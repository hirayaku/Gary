#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <type_traits>
#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "Defines.h"
#include "Utils.h"

enum SerDesVersion {
  V1 = 1,
};

enum GraphFormat {
  CSR,
  COO,
};

template <typename T>
struct Some {
  T ele;
  ~Some() { std::cout << "Destroyed: " << ele << std::endl; }
};

class GraphCSR;

class GraphCOO {
public:
  using VType = vidT;
  using VRange = utils::IdRange<vidT>;
  using EType = std::tuple<vidT, vidT>;
  static constexpr auto format = GraphFormat::COO;
  static int id;

  vidT numV() const { return m; }
  eidT numE() const { return nnz; }
  EType getEdge(eidT eid) const { return {row->at(eid), col->at(eid)}; }

  class ERange {
  public:
    struct Iter {
      public:
        Iter(const ERange &erange, eidT eid)
        : erange(erange), eid(eid) {}
        EType operator*() const { return {erange.row[eid], erange.col[eid]}; }
        bool operator==(const Iter &other) const { return eid == other.eid; }
        bool operator!=(const Iter &other) const { return eid != other.eid; }
        Iter operator++(int) {
          Iter old = *this;
          ++eid;
          return old;
        }
        Iter &operator++() {
          ++eid;
          return *this;
        }
      private:
        const ERange &erange;
        eidT eid;
    };  // struct Iter

    ERange(const GraphCOO &graph, eidT minId, eidT maxId);
    Iter begin() const { return Iter(*this, min); }
    Iter end() const { return Iter(*this, max); }

  private:
    const std::vector<vidT> &row, &col;
    eidT min, max;
  };

  VRange V() const { return VRange(0, m); }
  ERange E() const { return edgeBetween(0, nnz); }
  ERange edgeBetween(eidT e1, eidT e2) const { return ERange(*this, e1, e2); }

  GraphCOO() = default;

  template <typename RowT, typename ColT>
  GraphCOO(vidT m, eidT nnz, RowT &&row, ColT &&col)
  : m(m), nnz(nnz), row(std::forward<RowT>(row)), col(std::forward<ColT>(col))
  {}

  // template <typename RowT, typename ColT,
  //           std::enable_if_t<is_shared_ptr<RowT>::value, bool> = false,
  //           std::enable_if_t<is_shared_ptr<ColT>::value, bool> = false>
  // GraphCOO(vidT m, eidT nnz, RowT &&row, ColT &&col) 
  // : m(m), nnz(nnz),
  //   row(std::make_shared<std::vector<vidT>>(std::forward<RowT>(row))),
  //   col(std::make_shared<std::vector<vidT>>(std::forward<ColT>(col)))
  // {}

  // reverse the direction of edges
  void reverse_() {
    std::swap(row, col);
  }

  // reverse the direction of edges
  GraphCOO reverse() const {
    return GraphCOO(m, nnz, col, row);
  }

  // For each (u,v), create a copy of (v,u)
  // - for a directed graph, this produces an undirected graph
  // - for an undirected symmetrized graph, this produces an unsymmetrized copy 
  // - for an undirected unsymmetrized graph, this operation is invalid
  GraphCOO unsymmetrize() const;

  // convert graph COO to CSR (affects COO)
  GraphCSR toCSR_();

  // convert graph COO to CSR (immutable)
  GraphCSR toCSR() const;

  // serialization with cereal
  template <class Archive>
  void save(Archive &archive) const {
    archive(SerDesVersion::V1, format, m, nnz, row, col);
  }

  // deserialization with cereal
  template <class Archive>
  void load(Archive &archive) {
    SerDesVersion version;
    GraphFormat load_format;
    archive(version, load_format);
    std::cout << "Version:" << version << std::endl;
    if (load_format != format)
      throw cereal::Exception("Incompatible format: attempting to deserialize non-COO binary into GraphCOO");
    archive(m, nnz, row, col);
  }

protected:
  vidT m = 0;
  eidT nnz = 0;
  std::shared_ptr<std::vector<vidT>> row;
  std::shared_ptr<std::vector<vidT>> col;
};

class GraphCSR {
public:
  using VType = vidT;
  using VRange = utils::IdRange<VType>;
  using EType = std::tuple<vidT, vidT>;
  static constexpr auto format = GraphFormat::CSR;

  vidT numV() const { return m; }
  eidT numE() const { return nnz; }
  vidT getDegree(vidT v) const { return deg->at(v); }
  const auto &getDegreeAll() const { return *deg; }
  // vid: the source vertex ID of the requested edge
  // eno: the index of the edge in the adjacency list of vid; requires 0 <= eno < degree(vid)
  EType getEdge(vidT vid, vidT eno = 0) const { return {vid, idx->at(ptr->at(vid)+eno)}; }

  class ERange {
  public:
    struct Iter {
    public:
      Iter(const ERange &erange, vidT vid)
      : erange(erange), vid(vid), nid(0), eid(erange.rawPtr[vid])
      { nextVertex(); }
      EType operator*() const { return {vid, erange.rawIdx[erange.rawPtr[vid]+nid]}; }
      bool operator==(const Iter &other) const { return eid == other.eid; }
      bool operator!=(const Iter &other) const { return eid != other.eid; }
      Iter operator++(int) {
        Iter old = *this;
        ++nid; ++eid;
        nextVertex();
        return old;
      }
      Iter &operator++() {
        ++nid; ++eid;
        nextVertex();
        return *this;
      }
    private:
      const ERange &erange;
      vidT vid, nid;
      eidT eid;
      // if the neighbors of the current vertex are visited, jump to the next vertex
      inline void nextVertex() {
        // a vertex could have no neighbors
        while (eid >= erange.rawIdx[vid+1] && eid < erange.eidMax) {
          ++vid;
          nid = 0;
        }
      }
    }; // struct Iter

    // neighbor range of a single vertex
    ERange(const GraphCSR &graph, vidT vid);

    // neighbor range of multiple consecutive vertices
    ERange(const GraphCSR &graph, vidT vidMin, vidT vidMax);

    Iter begin() const { return Iter(*this, vidMin); }
    Iter end() const { return Iter(*this, vidMax); }
  private:
    const std::vector<eidT> &rawPtr;
    const std::vector<vidT> &rawIdx;
    vidT vidMin = 0, vidMax = 0;
    eidT eidMax = 0;
  };

  VRange V() const { return VRange(0, m); }

  ERange E() const { return adjBetween(0, m); }

  ERange adjBetween(vidT v1, vidT v2) const { return ERange(*this, v1, v2); }

  // get neighbors of node `vid`
  utils::Span<std::vector<vidT>::const_iterator> N(vidT vid) const;

  GraphCSR() = default;

  template <typename RowPtrT, typename ColIdxT>
  GraphCSR(vidT m, eidT nnz, RowPtrT &&ptr, ColIdxT &&idx)
  : m(m), nnz(nnz), ptr(std::forward<RowPtrT>(ptr)), idx(std::forward<ColIdxT>(idx)) {
    this->deg = std::make_shared<std::vector<vidT>>(numV());
    std::transform(this->ptr->begin()+1, this->ptr->end(), this->ptr->begin(),
      this->deg->begin(), std::minus<eidT>());
  }

  const std::vector<eidT> &getPtr() const { return *ptr; }

  const std::vector<vidT> &getIdx() const { return *idx; }

  // reverse the direction of edges
  void reverse_();

  // CSR -> CSC
  GraphCSR reverse() const;

  // For each (u,v), create a copy of (v,u)
  // - for a directed graph, this produces an undirected graph
  // - for an undirected symmetrized graph, this produces an unsymmetrized copy 
  // - for an undirected unsymmetrized graph, this operation is invalid
  GraphCSR unsymmetrize() const;

  // CSR -> COO
  GraphCOO toCOO() const;

  // serialization with cereal
  template <class Archive>
  void save(Archive &archive) const {
    archive(SerDesVersion::V1, format, m, nnz, ptr, idx, deg);
  }

  // deserialization with cereal
  template <class Archive>
  void load(Archive &archive) {
    SerDesVersion version;
    GraphFormat load_format;
    archive(version, load_format);
    if (load_format != format)
      throw cereal::Exception("Incompatible format: attempting to deserialize non-CSR binary into GraphCSR");
    archive(m, nnz, ptr, idx, deg);
  }

protected:
  vidT m = 0;
  eidT nnz = 0;
  std::shared_ptr<std::vector<eidT>> ptr;
  std::shared_ptr<std::vector<vidT>> idx;
  std::shared_ptr<std::vector<vidT>> deg;
};

// TODO: a partition of graph
class GraphPartitionCSR {
};

// load graph from the binary file
template <class GraphT>
GraphT loadFromBinary(std::string binFileName) {
  GraphT graph;
  std::ifstream input(binFileName, std::ios::binary);
  cereal::BinaryInputArchive iarchive(input);
  iarchive(graph);
  return graph;
}

// save graph to the binary file
template <class GraphT>
void saveToBinary(const GraphT &graph, std::string binFileName) {
  std::ofstream output(binFileName, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(graph);
}

// load graph datasets from SNAP: https://snap.stanford.edu/data/index.html
// assuming vertex ids start consecutively from 0
GraphCOO loadFromSNAP(std::string txtFileName);
