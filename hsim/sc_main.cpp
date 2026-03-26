#include <components/Top.h>
Configurations cfgs;

int sc_main(int argc, char* argv[])
{
    try {
        cfgs.init_configurations();

        const double host_freq = cfgs.get_host_freq();

        if (host_freq <= 0.0) {
            throw std::runtime_error("Invalid host frequency (must be > 0).");
        }

        Top top("top");

        sc_clock clk("clk", sc_time(1.0 / host_freq, SC_NS));
        top.clock(clk);

        const clock_t start_clock = clock();
        sc_start();
        const clock_t end_clock = clock();

        cfgs.print_configurations();

        std::cout << "Simulation Time : "
                  << std::setprecision(5)
                  << static_cast<double>(end_clock - start_clock) / CLOCKS_PER_SEC
                  << " (seconds)\n";
        std::cout << "Simulated Cycle : "
                  << static_cast<int>(sc_time_stamp().to_double() / 1000 / (1.0 / host_freq))
                  << " (cycles)\n";
        std::cout << "Simulated Time  : "
                  << static_cast<int>(sc_time_stamp().to_double() / 1000)
                  << " (ns)\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
