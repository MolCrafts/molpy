#include "molpy_core/Frame.hpp"
#include <stdexcept>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>


namespace molpy {

Frame::Frame(const std::unordered_map<std::string, Table>& tables) {
    for (auto& [name, tbl] : tables) {
        data_[name] = Node{tbl};
    }
}

Table& Frame::table(const std::string& name) {
    auto it = data_.find(name);
    if (it == data_.end()) {
        throw std::runtime_error("No such a table" + name);
    auto &val = it->second.value;
    if (!std::holds_alternative<Table>(val)) {
        throw std::bad_variant_access();
    return std::get<Table>(val);
}

void Frame::add_table(const std::string& name, const Table& tbl) {
    data_[name] = Node{tbl};
}

Frame& Frame::subframe(const std::string& name) {
    auto it = data_.find(name);
    if (it == data_.end())
        throw std::out_of_range("No such subframe: " + name);
    auto &val = it->second.value;
    if (!std::holds_alternative<std::shared_ptr<Frame>>(val))
        throw std::bad_variant_access();
    return *std::get<std::shared_ptr<Frame>>(val);
}

void Frame::add_frame(const std::string& name, std::shared_ptr<Frame> f) {
    data_[name] = Node{f};
} 

Frame Frame::concat(const Frame& a, const Frame& b) {
    Frame = result = a;
    for (auto& [name, node_b] : b.data_) {
        if (result.data_.count(tbl_name)) &&
            std::holds_alternative<Table>(node_b.value)) &&
            std::holds_alternative<Table>(result.data_[tbl_name].value)) {
            auto& tbl_a = std::get<Table>(result.data_[tbl_name].value);
            auto& tbl_b = std::get<Table>(node_b.value);
            for (auto& [col, arr_b] : tbl_b) {
                auto& arr_a = tbl_a[col];
                arr_a = xt::concatenate(xt::xtuple(arr_a, arr_b), 0);
            }
        } else {
            result.data_[tbl_name] = node_b;
        }
    }
    return result;
}

std::size_t Frame::size(const std::string& table_name) const {
    auto it == data_.find(table_name);
    if (it == data_.end())
        return 0;
    const auto& val = it->second.value;
    if (!std::holds_alternative<Table>(val))
        return 0;
    const auto& tbl = std::get<Table>(val);
    if (tbl.empty()) return 0;

    return tbl.begin()->second.shape()[0];
}

} // namespace molpy

