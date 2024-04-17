#include <iostream>
#include <cassert>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <fmt/format.h>
#include "HelperString.h"
#include "Graph.h"

void convertSNAP2Bin(std::string input, std::string output) {
  GraphCOO graph = loadFromSNAP(input);
  std::cout << fmt::format("{}: {} nodes, {} edges\n", input, graph.numV(), graph.numE());
  saveToBinary(graph, output);
}

void convertCOO2CSR(std::string input, std::string output) {
  GraphCOO graph = loadFromBinary<GraphCOO>(input);
  std::cout << fmt::format("{}: {} nodes, {} edges\n", input, graph.numV(), graph.numE());
  GraphCSR csr = graph.toCSR();
  auto &degrees = csr.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  saveToBinary(csr, output);
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", output, csr.numV(), csr.numE(), dmax);

  // check adjs are all sorted
  for (vidT vi : csr.V()) {
    vidT max = 0;
    for (vidT ni : csr.N(vi)) {
      assert((ni >= max && "adj not sorted"));
      max = ni;
    }
  }
}

void convertCSR2CSC(std::string input, std::string output) {
  GraphCSR graph = loadFromBinary<GraphCSR>(input);
  {
  auto &degrees = graph.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", input, graph.numV(), graph.numE(), dmax);
  }
  graph.reverse_();
  saveToBinary(graph, output);
  {
  auto &degrees = graph.getDegreeAll();
  auto dmax = *std::max_element(degrees.begin(), degrees.end());
  std::cout <<
    fmt::format("{}: {} nodes, {} edges, d_max={}\n", output, graph.numV(), graph.numE(), dmax);
  }
}

int main(int argc, const char *argv[])
{
  char *filename = nullptr;
  if (getCmdLineArgumentString(argc, argv, "input", &filename) == false) {
    throw std::invalid_argument("Cannot find the graph input location from command line");
  }
  std::string input(filename);
  if (getCmdLineArgumentString(argc, argv, "output", &filename) == false) {
    throw std::invalid_argument("Cannot find the output location from command line");
  }
  std::string output(filename);

  char *convert_flag = nullptr;
  std::string convert;
  if (getCmdLineArgumentString(argc, argv, "convert", &convert_flag))
    convert = convert_flag;

  const char *choices[] = { "snap2bin", "coo2csr", "csr2csc" };
  if (convert == choices[0]) {
    convertSNAP2Bin(input, output);
  } else if (convert == choices[1]) {
    convertCOO2CSR(input, output);
  } else {
    convertCSR2CSC(input, output);
  }

  return 0;
}
