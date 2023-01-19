#ifndef Random_H
#define Random_H

#include <random>

namespace molpy
{
    template<typename T>
    class uniform
    {
        public:
            uniform(T min, T max, unsigned int seed = std::random_device{}()): generator{seed}, dist{min, max} { }

            void setSeed(unsigned int seed) { };
            void setRange(T, T) {};

            T operator()() { return dist(generator); }

        private:
            std::mt19937 generator;
            std::uniform_real_distribution<T> dist;

    };

    template<>
    class uniform<int>
    {
        public:
            uniform(int min, int max, unsigned int seed = std::random_device{}()): generator{seed}, dist{min, max} { }

            void setSeed(unsigned int seed) { generator.seed(seed); }
            void setRange(int min, int max) { dist = std::uniform_int_distribution<int>{min, max}; }

            int operator()() { return dist(generator); }

        private:
            std::mt19937 generator;
            std::uniform_int_distribution<int> dist;

    };

    template<>
    class uniform<float>
    {
        public:
            uniform(float min, float max, unsigned int seed = std::random_device{}()): generator{seed}, dist{min, max} { }

            void setSeed(unsigned int seed) { generator.seed(seed); }
            void setRange(float min, float max) { dist = std::uniform_real_distribution<float>{min, max}; }

            float operator()() { return dist(generator); }

        private:
            std::mt19937 generator;
            std::uniform_real_distribution<float> dist;

    };
    
    template<>
    class uniform<double>
    {
        public:
            uniform(double min, double max, unsigned int seed = std::random_device{}()): generator{seed}, dist{min, max} { }

            void setSeed(unsigned int seed) { generator.seed(seed); }
            void setRange(double min, double max) { dist = std::uniform_real_distribution<double>{min, max}; }

            double operator()() { return dist(generator); }

        private:
            std::mt19937 generator;
            std::uniform_real_distribution<double> dist;

    };
    
    template<>
    class uniform<bool>
    {
        public:
            uniform(unsigned int seed = std::random_device{}()): generator{seed}, dist{0, 1} { }

            void setSeed(unsigned int seed) { generator.seed(seed); }

            bool operator()() { return dist(generator); }

        private:
            std::mt19937 generator;
            std::uniform_int_distribution<int> dist;

    };

};
#endif  // Random_H