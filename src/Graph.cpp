#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <atomic>
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

GraphCOO GraphCOO::unsymmetrize() const {
  GraphCOO cooT = this->reverse();
  std::vector<vidT> newrow(numE() * 2);
  std::vector<vidT> newcol(numE() * 2);
  std::copy(this->row->begin(), this->row->end(), newrow.begin());
  std::copy(this->col->begin(), this->col->end(), newcol.begin());
  std::copy(cooT.row->begin(), cooT.row->end(), newrow.begin()+numE());
  std::copy(cooT.col->begin(), cooT.col->end(), newcol.begin()+numE());
  return GraphCOO(
    numV(), numE() * 2,
    std::make_shared<std::vector<vidT>>(std::move(newrow)),
    std::make_shared<std::vector<vidT>>(std::move(newcol))
  );
}

GraphCSR GraphCOO::toCSR_()  {
  utils::radixSortInplacePar(*row, *col);

  std::vector<vidT> degreeTable(numV());
  std::vector<eidT> rowptr(numV()+1);
  #pragma omp parallel for
  for (size_t ei = 0; ei < row->size(); ++ei) {
    utils::atomicAdd(degreeTable[row->at(ei)], 1);
  }
  utils::prefixSumPar(numV(), degreeTable.data(), rowptr.data());
  assert(rowptr[numV()] == numE());

  #pragma omp parallel for
  for (vidT vi = 0; vi < numV(); ++vi) {
    std::sort(col->begin() + rowptr[vi], col->begin() + rowptr[vi+1]);
  }

  return GraphCSR(
    numV(), numE(),
    std::make_shared<std::vector<eidT>>(std::move(rowptr)), col
  );
}

GraphCSR GraphCOO::toCSR() const {
  GraphCOO coo {
    numV(), numE(),
    std::make_shared<std::vector<vidT>>(*this->row),
    std::make_shared<std::vector<vidT>>(*this->col)
  };
  return coo.toCSR_();
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

utils::Span<std::vector<vidT>::const_iterator> GraphCSR::N(vidT vid) const {
  // return ERange(*this, vid);
  return utils::Span<std::vector<vidT>::const_iterator>(
    idx->begin() + ptr->at(vid), deg->operator[](vid));
}

void GraphCSR::reverse_() {
  auto csc = reverse();
  std::swap(*this, csc);
}

GraphCSR GraphCSR::reverse() const {
  return this->toCOO().reverse().toCSR();
}

GraphCSR GraphCSR::unsymmetrize() const {
  return this->toCOO().unsymmetrize().toCSR_();
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

GraphCOO loadFromSNAP(std::string txtFileName) {
  FILE *fp = fopen(txtFileName.data(), "r");
  if (fp == nullptr)
    throw std::logic_error(fmt::format("Invalid SNAP file: {}", txtFileName));

  vidT maxId = 0;
  std::vector<vidT> row, col;

  char line[1024] = {'\0'};
  size_t lineno = 1;
  while(std::fgets(line, sizeof(line), fp) != nullptr) {
    if (line[sizeof(line)-2] != '\0')
      throw std::runtime_error(
        fmt::format("The line is too long ({}:{})", txtFileName, lineno));
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
