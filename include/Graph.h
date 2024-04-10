#pragma once

#include <vector>
#include <tuple>
#include <memory>
#include <type_traits>
#include "Defines.h"
#include "Utils.h"

// TODO:
// 0. Loader
// 1. COO <--> CSR <--> CSC
// 2. Partitioning

class GraphCSR;

class GraphCOO {
public:
  using VType = vidT;
  using VRange = utils::IdRange<vidT>;
  using EType = std::tuple<vidT, vidT>;

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
  GraphCOO(vidT m, eidT nnz, RowT row, ColT col)
  : m(m), nnz(nnz), row(row), col(col)
  {}

  void reverse_() {
    std::swap(row, col);
  }

  GraphCOO reverse() const {
    return GraphCOO(m, nnz, col, row);
  }

  GraphCSR toCSR() const;

  static GraphCSR load(std::string binFileName);

  static void save(std::string binFileName);

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

  vidT numV() const { return m; }
  eidT numE() const { return nnz; }
  vidT getDegree(vidT v) const { return ptr->at(v+1) - ptr->at(v); } 
  const auto &getDegree() const { return *ptr; }
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

  auto N(vidT vid) const {
    return utils::Span<std::vector<vidT>::const_iterator>(
      idx->begin() + ptr->at(vid), this->getDegree(vid));
  }

  GraphCSR() = default;

  template <typename RowPtrT, typename ColIdxT>
  GraphCSR(vidT m, eidT nnz, RowPtrT ptr, ColIdxT idx)
  : m(m), nnz(nnz), ptr(ptr), idx(idx) {}

  const std::vector<eidT> &getPtr() const { return *ptr; }

  const std::vector<vidT> &getIdx() const { return *idx; }

  void reverse_();

  GraphCSR reverse();

  GraphCOO toCOO() const;

  static GraphCSR load(std::string binFileName);

  static void save(std::string binFileName);

protected:
  vidT m = 0;
  eidT nnz = 0;
  std::shared_ptr<std::vector<eidT>> ptr;
  std::shared_ptr<std::vector<vidT>> idx;
};

// load graph from SNAP text format
// assuming vertex ids start consecutively from 0
GraphCOO loadFromSNAP(std::string txtFileName);
