#pragma once

#include "config.hpp"
#include "decomp.hpp"
#include "field.hpp"
#include "grid.hpp"
#include <memory>
#include <string>

class InitialData {
public:
    virtual ~InitialData() = default;
    virtual std::string name() const = 0;
    virtual void apply(const Params& p, const Grid& grid, const Decomp& decomp, State& state) const = 0;
};

std::unique_ptr<InitialData> make_initial_data(InitKind kind);
