#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <fmt/format.h>
#include "Graph.h"
#include "ParUtils.h"
#include "ParRadixSort.h"

GraphCOO::ERange::ERange(const GraphCOO &graph, eidT minId, eidT maxId)
: row(*graph.row), col(*graph.col), min(minId), max(maxId)
{
  if (minId > maxId || minId < 0 || maxId > graph.numE())
    throw std::invalid_argument(
      fmt::format("Invalid edge range ({}, {})", minId, maxId)
    );
}

GraphCOO GraphCOO::symmetrize() const {
  GraphCOO cooT = this->reverse();
  std::vector<vidT> newRow(numE() * 2);
  std::vector<vidT> newCol(numE() * 2);
  std::copy(this->row->begin(), this->row->end(), newRow.begin());
  std::copy(this->col->begin(), this->col->end(), newCol.begin());
  std::copy(cooT.row->begin(), cooT.row->end(), newRow.begin()+numE());
  std::copy(cooT.col->begin(), cooT.col->end(), newCol.begin()+numE());
  return GraphCOO {
    numV(), numE() * 2,
    std::make_shared<std::vector<vidT>>(std::move(newRow)),
    std::make_shared<std::vector<vidT>>(std::move(newCol))
  };
}

GraphCOO GraphCOO::orientation() const {
  const auto edges = E();
  eidT newNumE = std::count_if(edges.begin(), edges.end(), [](EType edge) {
    return std::get<0>(edge) > std::get<1>(edge);
  });
  std::vector<vidT> newRow(newNumE), newCol(newNumE);

  // thrust tuple is incompatible with std::tuple
  // std::copy_if(edges.begin(), edges.end(), thrust::make_zip_iterator(newRow.begin(), newRow.end()),
  //   [](EType edge) { return std::get<0>(edge) > std::get<1>(edge); }
  // );
  auto rowIter = newRow.begin();
  auto colIter = newCol.begin();
  for (const auto &e : edges) {
    vidT src, dst;
    std::tie(src, dst) = e;
    if (src > dst) {
      *rowIter++ = src;
      *colIter++ = dst;
    }
  }

  return GraphCOO {
    numV(), newNumE,
    std::make_shared<std::vector<vidT>>(std::move(newRow)),
    std::make_shared<std::vector<vidT>>(std::move(newCol))
  };
}

GraphCSR GraphCOO::toCSR_(bool sorted)  {
  if (!sorted)
    utils::radixSortInplacePar(*row, *col);

  std::vector<vidT> degreeTable(numV());
  std::vector<eidT> rowptr(numV()+1);
  #pragma omp parallel for
  for (size_t ei = 0; ei < row->size(); ++ei) {
    utils::atomicAdd(degreeTable[row->at(ei)], 1);
  }
  utils::prefixSumPar(numV(), degreeTable.data(), rowptr.data());
  if(rowptr[numV()] != numE()) {
    throw std::logic_error(
      "Inconsistent rowptr when converting COO to CSR"
    );
  }

  if (!sorted) {
    #pragma omp parallel for schedule(dynamic)
    for (vidT vi = 0; vi < numV(); ++vi) {
      std::sort(col->begin() + rowptr[vi], col->begin() + rowptr[vi+1]);
    }
  }

  return GraphCSR(
    numV(), numE(),
    std::make_shared<std::vector<eidT>>(std::move(rowptr)), col
  );
}

GraphCSR GraphCOO::toCSR(bool sorted=false) const {
  GraphCOO coo {
    numV(), numE(),
    std::make_shared<std::vector<vidT>>(*this->row),
    std::make_shared<std::vector<vidT>>(*this->col)
  };
  return coo.toCSR_(sorted);
}

GraphCSR::ERange::ERange(const GraphCSR &graph, vidT vid)
: rawPtr(*graph.ptr), rawIdx(*graph.idx)
{
  if (vid < 0 || vid >= graph.m)
    throw std::invalid_argument(fmt::format("Invalid vertex %ld", vid));
  this->vidMin = vid;
  this->vidMax = vid + 1;
  this->eidMax = rawPtr[vidMax];
}

GraphCSR::ERange::ERange(const GraphCSR &graph, vidT vidMin, vidT vidMax)
: rawPtr(*graph.ptr), rawIdx(*graph.idx)
{
  if (vidMin > vidMax || vidMin < 0 || vidMax >= graph.m)
    throw std::invalid_argument(fmt::format("Invalid vertices %ld, %ld", vidMin, vidMax));
  this->vidMin = vidMin;
  this->vidMax = vidMax;
  this->eidMax = rawPtr[vidMax];
}

utils::Span<vidT> GraphCSR::N(vidT vid) const {
  // return ERange(*this, vid);
  return utils::Span<vidT>(
    idx->data() + ptr->at(vid), deg->operator[](vid));
}

void GraphCSR::reverse_() {
  auto csc = reverse();
  std::swap(*this, csc);
}

GraphCSR GraphCSR::reverse() const {
  return this->toCOO().reverse().toCSR();
}

GraphCSR GraphCSR::symmetrize() const {
  return this->toCOO().symmetrize().toCSR_();
}

GraphCOO GraphCSR::toCOO() const {
    std::vector<vidT> row(this->numE());
    auto numV = this->numV();
    auto &ptr = *this->ptr;

    #pragma omp parallel for
    for (vidT v = 0; v < numV; ++v) {
      std::fill(row.begin()+ptr[v], row.begin()+ptr[v+1], v);
    }

    return GraphCOO(
        numV, this->numE(),
        std::make_shared<std::vector<vidT>>(std::move(row)), this->idx
    );
}

GraphCOO loadFromSNAP(fs::path txtFileName, bool removeSelfLoops) {
  FILE *fp = fopen(txtFileName.c_str(), "r");
  if (fp == nullptr)
    throw std::logic_error(fmt::format("Invalid SNAP file: {}", txtFileName.string()));

  vidT maxId = 0;
  std::vector<vidT> row, col;

  char line[1024] = {'\0'};
  size_t lineno = 1;
  while(std::fgets(line, sizeof(line), fp) != nullptr) {
    if (line[sizeof(line)-2] != '\0')
      throw std::runtime_error(
        fmt::format("The line is too long ({}:{})", txtFileName.string(), lineno));
    ++lineno;

    // skip comments and empty lines
    unsigned sptr = 0;
    while(sptr < sizeof(line) && std::isspace(line[sptr])) sptr++;
    if (line[sptr] == '\0') continue;
    if (line[sptr] == '#') continue;

    // expect vertex ids;
    char *pos;
    vidT src = utils::str2int<vidT>(&line[sptr], &pos, 10);
    vidT dst = utils::str2int<vidT>(pos, nullptr, 10);
    if (removeSelfLoops && src == dst) continue;
    if (src > maxId) maxId = src;
    if (dst > maxId) maxId = dst;
    row.push_back(src);
    col.push_back(dst);
  }

  vidT numVertex = maxId + 1;
  eidT numEdge = row.size();
  return GraphCOO(numVertex, numEdge,
    std::make_shared<std::vector<vidT>>(std::move(row)),
    std::make_shared<std::vector<vidT>>(std::move(col))
    );
}
