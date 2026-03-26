#include "pim_wrapper.h"
#include <algorithm>
#include <cmath>
#include <vector>

PIMWrapper::PIMWrapper(sc_module_name name, Logger* logger)
: slave("slave"), clock("clock"), logger(logger), peq(this, &PIMWrapper::peq_cb)
{
	if (cfgs.get_memory_structure() == "partitioned") {
		SC_THREAD(simulate_partitioned_pim);
	} else if (cfgs.get_memory_structure() == "unified") {
		SC_THREAD(simulate_unified_pim);
	}

	SC_METHOD(clock_posedge);
    sensitive << clock.pos();
	dont_initialize();

	SC_METHOD(clock_negedge);
    sensitive << clock.neg();
	dont_initialize();

	slave.register_nb_transport_fw(this, &PIMWrapper::nb_transport_fw);

	mode_change_delay = cfgs.get_mode_change_delay();
    pim_sync_map = cfgs.get_pim_sync_map();
	host_sync_map = cfgs.get_host_sync_map();

	init_pim_dram();
}

PIMWrapper::~PIMWrapper()
{
	if (bridge) {
		bridge->print_stats();
		delete bridge;
	}
}

void PIMWrapper::init_pim_dram()
{
	std::string type = cfgs.get_pim_type();
	std::string config = cfgs.get_pim_config();
	std::string preset = cfgs.get_pim_preset();
	uint32_t channel_num = cfgs.get_pim_channels();

	std::string path(DRAMSIM3_PATH);
	std::string cfg_path = path + "configs/" + type + "/" + config + "_" + preset
								+ "_ch" + std::to_string(channel_num) + ".ini";
	bridge = new dramsim3::Bridge(cfg_path.c_str(), path.c_str());
}

void PIMWrapper::simulate_partitioned_pim()
{
	double period = 1/cfgs.get_pim_freq();

	while (1) {
		/* Check completed transactions from bridge */
        if (auto* completed_trans = bridge->getCompletedCommand()) {  
			if (completed_trans->get_command() == TLM_PIM_COMMAND) {		
				compute_completed_queue.push_back(completed_trans);
				if (compute_outstanding > 0) --compute_outstanding;
			} else {
				memory_completed_queue.push_back(completed_trans);	
				if (memory_outstanding > 0) --memory_outstanding;
			}
		}

		if (memory_outstanding == 0 && !compute_pending_queue.empty()) {
			auto* trans = compute_pending_queue.front();
			compute_pending_queue.pop_front();
			send_transaction(trans, true);
			++compute_outstanding;
		}
		else if (compute_outstanding == 0 && !memory_pending_queue.empty()) {
			auto* trans = memory_pending_queue.front();
			memory_pending_queue.pop_front();
			send_transaction(trans, false);
			++memory_outstanding;
		}

		bridge->ClockTick();

		/* Check termination condition */
		if (active_host == 0 &&
			active_core == 0 &&
			compute_outstanding == 0 &&
			memory_outstanding == 0 &&
			compute_pending_queue.empty() &&
			memory_pending_queue.empty() &&
			compute_completed_queue.empty() &&
			memory_completed_queue.empty()) {
				active_dram = 0;
				break;
			}

		/* Wait for the next PIM clock cycle */
		wait(period, SC_NS);
	}
}

void PIMWrapper::simulate_unified_pim()
{
	double period = 1/cfgs.get_pim_freq();

	while (1) {
		/* Check completed memory transactions */
        if (auto* completed_trans = bridge->getCompletedCommand()) {
			if (completed_trans->get_command() == TLM_PIM_COMMAND) {
				compute_completed_queue.push_back(completed_trans);
				if (compute_outstanding > 0) {
					--compute_outstanding;
				}
			} else {
				memory_completed_queue.push_back(completed_trans);
				if (memory_outstanding > 0) {
					--memory_outstanding;
				}
			}
        }

		/* Process pending transactions */
        if (memory_outstanding == 0 && !compute_pending_queue.empty())
		{
            auto* trans = compute_pending_queue.front();
            compute_pending_queue.pop_front();
            send_transaction(trans, true);
            compute_outstanding++;
        }
		else if (compute_outstanding == 0 && !memory_pending_queue.empty())
		{
            auto* trans = memory_pending_queue.front();
            memory_pending_queue.pop_front();
            send_transaction(trans, false);
            memory_outstanding++;
        }

		bridge->ClockTick();

		/* Check termination condition */
			if (active_host == 0 &&
				active_core == 0 &&
				compute_outstanding == 0 &&
				memory_outstanding == 0 &&
					compute_pending_queue.empty() &&
					memory_pending_queue.empty() &&
					compute_completed_queue.empty() &&
					memory_completed_queue.empty()) {
						active_dram = 0;
						break;
					}

					/* Wait for the next PIM clock cycle */
		wait(period, SC_NS);
	}
}

void PIMWrapper::send_transaction(tlm::tlm_generic_payload* trans, bool is_compute)
{
    if (is_compute) {
        process_mode_transition(PimMode::COMPUTE);
        bridge->sendCommand(*trans);

		logger->update_start(trans->get_layer(), PIM);
    } else {
        process_mode_transition(PimMode::MEMORY);
        bridge->sendCommand(*trans);
    }
}

void PIMWrapper::process_mode_transition(PimMode target_mode)
{
	if (pim_mode != target_mode) {
		mode_change_count++;
		unsigned int  delay = (target_mode == PimMode::MEMORY)
			? std::get<0>(mode_change_delay)
			: std::get<1>(mode_change_delay);

		wait(sc_time(delay, SC_NS));
		pim_mode = target_mode;
	}		
}

void PIMWrapper::clock_posedge()
{
	if (memory_completed_queue.empty()) return;

	auto* completed_trans = memory_completed_queue.front();
	memory_completed_queue.pop_front();

	bool is_read =
		(completed_trans->get_command() == TLM_READ_COMMAND) ||
		(completed_trans->get_src_id() == NPU);

	if (is_read) {
		/* Read response to Host, Read/Write response to NPU */
		unsigned int dst_id = (completed_trans->get_src_id() == MCU) ? 1 : 0;
		send_response(completed_trans, dst_id);
	} else if (completed_trans->get_command()==TLM_WRITE_COMMAND) {
		/* Notify Host/MCU write completion */
		complete_write_request(completed_trans);
	}
}

void PIMWrapper::clock_negedge()
{	
	if (compute_completed_queue.empty()) return ;

	auto* trans = compute_completed_queue.front();
	compute_completed_queue.pop_front();

	const std::string& layer = trans->get_layer();

	if (trans->is_last()) {
		if (layer.find("RoPE") != std::string::npos || layer == "Mul1") {
			/* Computation end signal */
			auto src = trans->get_src_id();
			if (src == HOST) logger->update_end(layer, HOST);
			else if (src == MCU) logger->update_end(layer, MCU);
			is_pim_busy = false;
			pim_compute_done_ev.notify();
		}
		signal_sync(pim_sync_map, layer);
		logger->update_end(layer, PIM);
	}
}

void PIMWrapper::send_response(tlm::tlm_generic_payload* trans, unsigned int dst_id)
{	
	tlm_phase phase = BEGIN_RESP;
	tlm_sync_enum reply = slave[dst_id]->nb_transport_bw(*trans, phase, t);
	assert(reply == TLM_UPDATED);
}

void PIMWrapper::complete_write_request(tlm::tlm_generic_payload* trans)
{	
	const std::string& layer = trans->get_layer();
	BaseMap* base_map = (trans->get_pim_cmd() == "addertree") ? &pim_sync_map : &host_sync_map;
	if (trans->is_last()) {
		signal_sync(*base_map, layer);
		if (trans->get_src_id() == HOST) {
			logger->update_end(layer, HOST);
		}
		else if (trans->get_src_id() == MCU) {
			logger->update_end(layer, MCU);
		}
	}
	trans->release();
}

tlm_sync_enum PIMWrapper::nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
	peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void PIMWrapper::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    if (trans.get_command() == TLM_PIM_COMMAND) {
		compute_pending_queue.push_back(&trans);
    } else {
		memory_pending_queue.push_back(&trans);
    }
}

void PIMWrapper::signal_sync(BaseMap& map, const std::string& layer)
{
	auto it = map.find(layer);
	if (it != map.end()) {
		unsigned int sync_id = std::get<0>(it->second);
		std::cout << "[PIM][SIGNAL] " << layer << " (ID:" << sync_id << "), cycle:" << current_cycle() << "\n";
		pim_sync_obj.signal(sync_id);
	} else {
		std::cerr << "[PIM][ERROR] Key[" << layer << "] not found in the map!\n";
	}
}
