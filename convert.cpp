#include <iostream>
#include <cassert>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include "HelperString.h"
#include "Graph.h"

namespace fs = std::filesystem;

void convertSNAP2Bin(fs::path input, fs::path output, bool keepLoops) {
  GraphCOO graph = loadFromSNAP(input, !keepLoops);
  std::cout << fmt::format("{}: {} nodes, {} edges\n", input.string(), graph.numV(), graph.numE());
  saveToBinary(graph, output);
}

void convertCOO2CSR(fs::path input, fs::path output, bool symmetrize) {
  GraphCOO graph = loadFromBinary<GraphCOO>(input);
  std::cout << fmt::format("{}: {} nodes, {} edges\n", input.string(), graph.numV(), graph.numE());

  if (symmetrize)
    graph = graph.symmetrize();

  GraphCSR csr = graph.toCSR_();
  auto &degrees = csr.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  saveToBinary(csr, output);
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", output.string(), csr.numV(), csr.numE(), dmax);

  // check adjs are all sorted
  for (vidT vi : csr.V()) {
    vidT max = 0;
    for (vidT ni : csr.N(vi)) {
      assert((ni >= max && "adj not sorted"));
      max = ni;
    }
  }
}

void convertCSR2CSC(fs::path input, fs::path output) {
  GraphCSR graph = loadFromBinary<GraphCSR>(input);
  {
  auto &degrees = graph.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", input.string(), graph.numV(), graph.numE(), dmax);
  }
  graph.reverse_();
  saveToBinary(graph, output);
  {
  auto &degrees = graph.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", output.string(), graph.numV(), graph.numE(), dmax);
  }
}

void convertOrientCSR(fs::path input, fs::path output) {
  GraphCSR graph = loadFromBinary<GraphCSR>(input);
  {
  auto &degrees = graph.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", input.string(), graph.numV(), graph.numE(), dmax);
  }
  auto trimmed = graph.toCOO().orientation().toCSR_();
  saveToBinary(trimmed, output);
  {
  auto &degrees = trimmed.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n",
      input.string(), trimmed.numV(), trimmed.numE(), dmax);
  }
}

int main(int argc, const char *argv[])
{
  char *filename = nullptr;
  if (getCmdLineArgumentString(argc, argv, "input", &filename) == false) {
    throw std::invalid_argument("Cannot find the graph input location from command line");
  }
  fs::path input(filename);
  if (getCmdLineArgumentString(argc, argv, "output", &filename) == false) {
    throw std::invalid_argument("Cannot find the output location from command line");
  }
  fs::path output(filename);

  char *convert_flag = nullptr;
  std::string convert;
  if (getCmdLineArgumentString(argc, argv, "convert", &convert_flag))
    convert = convert_flag;

  int symmetrize = getCmdLineArgumentInt(argc, argv, "symmetrize");
  int keepLoops = getCmdLineArgumentInt(argc, argv, "keepLoops");

  const char *choices[] = { "snap2bin", "coo2csr", "csr2csc", "orient" };
  if (convert == choices[0]) {
    convertSNAP2Bin(input, output, keepLoops);
  } else if (convert == choices[1]) {
    convertCOO2CSR(input, output, symmetrize);
  } else if (convert == choices[2]) {
    convertCSR2CSC(input, output);
  } else if (convert == choices[3]) {
    convertOrientCSR(input, output);
  }

  return 0;
}
