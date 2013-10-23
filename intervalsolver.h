#ifndef INTERVALSOLVER_H
#define INTERVALSOLVER_H

// Really big number
#define INFINITY (INT_MAX/16)

#include "platformspecific.h"
#include <iostream>
#include <algorithm>

using namespace std;

/**
 * Iterate over the elements of a bitset
 */ 
class BitsetElementIterator {
public:
  BitsetElementIterator(unsigned int bitset) : seen(bitset) {
    element = smallestIndex(bitset);
  }

  bool operator==(BitsetElementIterator it2) { return seen == it2.seen; }
  bool operator==(unsigned int bitset) { return seen == bitset; }
  bool operator!=(BitsetElementIterator it2) { return seen != it2.seen; }
  bool operator!=(unsigned int bitset) { return seen != bitset; }

  BitsetElementIterator &operator++() {
    // Mark the element we have already seen
    seen    = seen & ~(1 << element);
    // Locate the next element
    element = smallestIndex(seen);
    return *this; 
  }

  int operator*() { return element; }
  
private:
  unsigned int seen; 
  int element;
};

/**
 * Represents a bitset.
 * Allows iteration over its values using a BitsetElementIterator.
 */
class Bits {
public:
  Bits(unsigned int bitset) : bitset(bitset) {}
  BitsetElementIterator begin() { return BitsetElementIterator(bitset); }
  //BitsetElementIterator end() { return BitsetElementIterator(0U); }
  unsigned int end() { return 0U; }
private:
  unsigned int bitset;
};

// Held-Karp-like algorithm to solve base cases
class IntervalSolver {
public:

  /**
   * Decide for a vertex i, which vertices j may not precede it on the path
   */
  unsigned int* mayNotPrecede(const int n, int **mayPrecede) {
    unsigned int *noPrecedence = new unsigned int[n];
    for (int i = 0; i < n; ++i) {
      unsigned int ni = 0U;
      for (int j = 0; j < n; ++j) {
        if (!mayPrecede[j][i]) {
          ni = ni | (1 << j);
        }
      }

      noPrecedence[i] = ni;
    }

    return noPrecedence;
  }


  /**
   * Checks whether or not a feasible solution exists
   */
  /*unsigned int *solveFeasible(const int n, int **transition, unsigned int start) {
    unsigned int *noTransition = mayNotPrecede(n, transition);

    unsigned int size = (1 << n) - 1;
    // Map each subset and each last vertex to the cost of an optimal path
    unsigned int *best = new unsigned int[size];

    for (unsigned int set = 0; set < size; ++set) {
      best[set] = ~0U;
    }

    // Initialisation of DP
    best[0] = start;
    
    // Main DP 
    for (unsigned int set = 1; set < size; ++set) {
      // Iterate over each value in the set
      unsigned int unset = set;
      while (unset) {
        // Let's visit j last in the subset, then go from j to k for all possible k
        int j = smallestIndex(unset);
        unset = unset & ~(1 << j);
        unsigned int costs  = noTransition[j]; // TODO: other way around?
        unsigned int subset = set & ~(1 << j);
        best[set] = best[set] & (best[subset] | costs); // TODO: set
        // Iterate over all values not contained in set
        unsigned int bubset = ~set & size;
      }
    }

    delete[] noTransition;
    return best;
  }*/

  /**
   * Solve the TSP for each subset of nodes  where some nodes may not preceed others.
   */
  int **solveSetsPrecedence(const int n, int **transition, int **mayPrecede, int *start) {
    unsigned int *noPrecedence = mayNotPrecede(n, mayPrecede);
    unsigned int size = (1 << n) - 1;
    // Map each subset and each last vertex to the cost of an optimal path
    int **best = new int*[size];

    for (unsigned int set = 0; set < size; ++set) {
      best[set] = new int[n];
      for (int i = 0; i < n; ++i) best[set][i] = INFINITY;
    }

    // Initialisation of DP
    for (int i = 0; i < n; ++i) {
      best[0][i] = start[i];
    }

    // Main DP 
    for (unsigned int set = 1; set < size; ++set) {
      // Iterate over each value in the set
      unsigned int unset = set;
      while (unset) {
        // Let's visit j last in the subset, then go from j to k for all possible k
        int j = smallestIndex(unset);
        unset = unset & ~(1 << j);
        unsigned int subset = set & ~(1 << j);
        // Iterate over all values not contained in set
        unsigned int bubset = ~set & size;
        while (bubset) {
          // Make sure we are not adding k after values that may not precede it
          int k = smallestIndex(bubset);
          // Will simply unset the last bit
          bubset = bubset & (bubset - 1);
          // Make sure nothing that may not precede k does
          if ((noPrecedence[k] & set) == 0U) {
            // Compute the minimum
            int around = best[subset][j] + transition[j][k];
            if (around < best[set][k]) best[set][k] = around;
          }
        }
      }
    }

    delete[] noPrecedence;

    return best;
  }

  /**
   * Solve the TSP for each subset of a graph.
   * n: size of the graph
   * transition: transition cost matrix of size at least n
   * start: cost to initially visit each node
   */
  int **solveSets(const int n, int **transition, int *start) {
    //unsigned int size = (1 << n) - 1;
    unsigned int size = (1 << n) - 1;
    // Map each subset and each last vertex to the cost of an optimal path
    int **best = new int*[size];

    for (unsigned int set = 0; set < size; ++set) {
      best[set] = new int[n];
      for (int i = 0; i < n; ++i) best[set][i] = INFINITY;
    }

    // Initialisation of DP
    for (int i = 0; i < n; ++i) {
      best[0U][i] = start[i];
    }

    if (n <= 1) return best;

    // Main DP 
    for (unsigned int set = 1; set < size; ++set) { 
      // Iterate over each value in the set
      unsigned int unset = set;
      while (unset) {
        // Let's visit j last instead, then go from j to k
        int j = smallestIndex(unset);
        unset = unset & ~(1 << j);
        unsigned int subset = set & ~(1 << j);
        // Iterate over all values not contained in set
        unsigned int bubset = ~set & size;
        while (bubset) {
          int k = smallestIndex(bubset);
          // Will simply unset the last bit
          bubset = bubset & (bubset - 1);
          // Compute the minimum
          int around = best[subset][j] + transition[j][k];
          if (around < best[set][k]) {
            best[set][k] = around;
          }
        }
      }
    }

    return best;
  }


  /**
   * Solve TSP for each subset of vertices
   * For each set S, the resulting array returns the best
   * path from start to goal containing all of the vertices in this set
   */
  int *solve(const int n, int **transition, int *start, int *goal) {
    return convert(solveSets(n, transition, start), n, goal);
  }


  /**
   * Given a selection, updates the path array by writing the elements
   * of this selection in their optimal order.
   * Returns a pointer to the place in path where the first element was written.
   * best: solution given by solveSets
   * transition: transition cost matrix
   * goal: transition cost to the final node
   * path: array that will contain result, must be of size at least |selection| 
   ° selection: selection for which to compute the optimal path
   */
  int *updatePath(int **best, int **transition, int *goal, int *path, unsigned int selection) {
    // Nothing is selected and thus needs to be updated
    if (selection == 0U) return path;

    int last = computeLastElement(selection, best, goal);
    --path;
    *path = last;

    return computePartialPath(best, transition, selection & ~(1 << last), path);
  }

  /**
   * Given a selection, updates the path array by writing the elements
   * of this selection in their optimal order.
   * Returns a pointer to the place in path where the first element was written.
   * best: solution given by solveSets
   * transition: transition cost matrix
   * path: array that will contain result, must be of size at least |selection| 
   ° selection: selection for which to compute the optimal path
   */
  int* computePartialPath(int **best, int **transition, unsigned int set, int *path) {
    int *p = path;

    // Iterate over each value in the set
    while (set) {
      unsigned int unset = set;
      int bestIndex = 0;
      int bestCost = INFINITY;

      // Search for the next value to add to the path
      while (unset) {
        int j = smallestIndex(unset);
        unset = unset & ~(1 << j);
        unsigned int partial = set & ~(1 << j);
        int cost = best[partial][j] + transition[j][*p];

        if (cost < bestCost) {
          bestIndex = j;
          bestCost  = cost;
        }
      }

      // Add it to the path
      --p;
      *p = bestIndex;
      set = set & ~(1 << bestIndex);
    }

    return p;
  }

  
  /**
   * Compute the last element of a path of the optimal solution
   * set: selection for which the opimal solution is being computed
   * best: result of solveSets functions
   * goal: vector 
   */
  int computeLastElement(unsigned int set, int** best, int *goal) {
    // All bits set to 1
    int bestIndex = 0;
    int bestCost  = INFINITY;
    unsigned int unset = set;
    while (unset) {
      // Let's visit j last instead, then go from j to i
      int j = smallestIndex(unset);
      unset = unset & ~(1 << j);
      unsigned int subset = set & ~(1 << j);

      int around = best[subset][j] + goal[j];
      if (around < bestCost) {
        bestCost  = around;
        bestIndex = j;
      } 
    }
    
    return bestIndex;
  }

  /**
   * Given the solution of solveSets, compute the actual best path.
   */
  int *computePath(const int n, int** best, int **transition, int *goal) {
    int *path = new int[n];
    return updatePath(best, transition, goal, path + n, (1 << n) - 1);
  }

  /**
   * Given the result mapping each subset of vertices and each end vertex
   * to the cost of an optimal path, compute for each subset the cost
   * of going to an end vertex, where the array goal represents the cost
   * of going from each vertex to the end vertex
   */
  int *convert(int **result, int n, int *goal) {
    unsigned int size = 1 << n;
    int *final = new int[size];

    // An empty set; an edge from start to goal is not allowed (by construction)
    final[0U] = INFINITY;

    /**
     * Compute the optimal sequence from start to goal
     * that visits everything in 'set'.
     */
    for (unsigned int set = 1; set < size; ++set) {
      int min = INFINITY;
      unsigned int unset = set;
      while (unset) {
        // Let's visit j last instead, then go from j to i
        int j = smallestIndex(unset);
        unset = unset & ~(1 << j);
        unsigned int subset = set & ~(1 << j);

        int x = result[subset][j] + goal[j];
        if (x < min) min = x;
      }

      final[set] = min;
    }

    return final;
  }


  /**
   * Cleans the result of the array returned by solveSets
   */
  void cleanResult(int **result, int n) {
    int size = (1 << n) - 1;
    for (int i = 0; i < size; ++i) {
      delete[] result[i];
    }
    delete[] result;
  }

};


class MultiIntervalSolver {
  // Total number of nodes 
  const int n;
  // Number of nodes in each interval
  const int k;
  // Transition cost matrix
  int **transition;
  // Can node i preceed node j in a good sequence
  int **precedence;
  // Partial solution for each step, used to compute the actual path
  int ***partial;
  // Solutions of the base cases, used to compute the actual path
  int ***parts;
  // Solver for each interval
  IntervalSolver solver;

public:

  MultiIntervalSolver(int n, int k, int **transition, int **precedence) 
    : n(n), k(k), transition(transition), precedence(precedence) {
    partial = new int**[n - k + 1];    
    parts   = new int**[n - k + 1];    
  }

  ~MultiIntervalSolver() {
    delete[] partial;
    delete[] parts;
  }

  
  /**
   * Compute the optimal path
   */
  int *solve() {
    int *initial = new int[k];
    for (int i = 0; i < k; ++i) initial[i] = 0;
    // First step: compute first segment 0 .. k - 1
    int **part = solver.solveSets(k, transition, initial);
    partial[0] = part;
    parts[0]   = part;

    delete[] initial;

    for (int step = 1; step + k <= n; ++step) {
      // Allocate partial matrix
      partial[step] = new int*[1 << k];
      for (int i = 0; i < (1 << k); ++i) {
         partial[step][i] = new int[k];
         for (int j = 0; j < k; ++j) {
           partial[step][i][j] = INFINITY;
         }
      }

      // Transition cost matrix for this part
      int **trans = submatrix(transition, step, k);
      // Costs to the final node of this part
      initial     = start(step + k - 1);
      // Compute for each subset and end node the optimal path cost
      parts[step] = solver.solveSets(k - 1, trans, initial);

      dynamic(partial[step - 1], partial[step], parts[step], trans);

      delete[] trans;
      delete[] initial;
      //cout << "Optimal step " << step  << " = " << partial[step][(1 << k) - 2][0] << endl;
      //solver.cleanResult(part, k - 1);
    }

    // Compute the optimal path
    int *path = computeSelections();
    //int *path = 0;

    solver.cleanResult(parts[0], k);
    for (int step = 1; step + k <= n; ++step) 
      solver.cleanResult(parts[step], k - 1);
    
    for (int step = 1; step + k <= n; ++step) {
      for (int i = 0; i < (1 << k); ++i) {
        delete[] partial[step][i];
      }
      delete[] partial[step];
    }

    return path;
  }

  /**
   * Compute the initial cost vector for an element x.
   * The transition costs to all its k predecessors are added.
   */ 
  int *start(int x) {
    int *s = new int[k];
    for (int i = 0; i < k; ++i) 
      s[i] = transition[x][x - k + i + 1];
    
    return s;
  }

  /**
   * Compute the ifnal cost vector for an element x.
   * The transition costs from all elements in the matrix are added.
   */
  int *goal(int **matrix, int end, int size) {
    int *g = new int[size];
    for (int i = 0; i < size; ++i) {
      g[i] = matrix[i][end];
    }
    
    return g;
  }

  /**
   * Compute the square submatrix s[start .. start + size - 1][start .. start + size - 1]
   * Contains pointers to the original matrix, and should thus be deleted using delete[] only.
   */
  int **submatrix(int **matrix, int start, int size) {
    int **sub = new int*[size];
    for (int i = 0; i < size; ++i) 
      sub[i] = matrix[start + i] + start;

    return sub;
  }


  /**
   *
   */
  int *computeSelections() {
    //int **sub = submatrix(transition, n - k - 1, k);
    unsigned int small = (1 << (k - 1)) - 1;
    int **sub = submatrix(transition, n - k, k);
    // Compute the final end node (TODO: correct index?)
    int end   = selectionNotContained(partial[n - k - 0], sub, small); 
    // Cost vector to end node
    int *g;
    // Compute the selection of the first piece
    unsigned int selection = ((1 << k) - 1) & ~(1 << end); 
    
    int *path = new int[n];
    for (int i = 0; i < n; ++i) path[i] = -1;
    int *p    = path + n;

    delete[] sub;

    cout << "Optimal computed: " << partial[n - k - 1][selection][end] << endl;
    for (int step = n - k; step > 0; --step) {
      // Compute relevant transition matrix
      sub = submatrix(transition, step, k);
      // Case 1) e is not selected or the final node
      unsigned int selected = 0U;
      int j = end;

      // Node e is in our selection, compute selected path
      if (selection > small) {
        // Case 3) e is selected
        j = selectionContained(partial[step - 1], parts[step], sub, selection, end, &selected);
        g = goal(sub, end, k);
        int *e = p;

        // The node we are working towards
        *(--p) =  end;
        // Compute the path from e to j
        p = solver.updatePath(parts[step], sub, g, p, selected);
        // Place e on the path
        *(--p) =  k - 1;
        // Compute the actual indices of the nodes on the path
        add(p, e, step);

        delete[] g;
      } else if (end == k - 1) {
        // Case 2) e is the final node
        *(--p) = k - 1 + step;
        j = selectionNotContained(partial[step - 1], sub, selection);
      }

      // Update selection for the next iteration
      selection = selection ^ selected;
      selection = ((selection & ~(1 << j)) << 1) | 1;
      selection = selection & ((1 << k) - 1);
      end = j + 1;

      delete[] sub;
    }

    sub = submatrix(transition, 0, k);
    g = goal(sub, end, k);
    *(--p) = end;
    p = solver.updatePath(partial[0], sub, g, p, selection);

    delete[] sub;
    delete[] g;
    /*
    for (int i = 0; i < n; ++i) cout << path[i] << " "; cout << endl;
    std::sort(path, path + n);
    for (int i = 0; i < n; ++i) cout << path[i] << " "; cout << endl;
    cout << "Path of length: " << (path + n - p) << endl;
    cout << "Should of course be of length: " << n << endl;
    */

    return path;
  }

  void add(int *start, int *end, int x) {
    while (start != end) {
      *start += x;
      ++start;
    }
  }

  /**
   * If e is contained in a part, compute the optimal selection
   * set: the set of nodes to use
   * j: final node where to compute a path to
   * selection: will contain the subset of set which is used in this part
   */ 
  int selectionContained(int **last, int **part, int **transition, unsigned int set, int j, unsigned int *selected) {
    int bestCost = INFINITY;
    int best     = 0;

    set &= ~(1 << (k - 1));
    // Assume e is contained in the set
    // TODO: which of these is correctish?
    for (unsigned int subset = set; subset; subset = (subset - 1) & set) {
    //for (unsigned int subset = (set - 1 ) & set; subset; subset = (subset - 1) & set) {
      Bits endsI(subset);
      for (BitsetElementIterator iti = endsI.begin(); iti != endsI.end(); ++iti) {
        // For each i as last index
        int i = *iti;
        // Index of subset in the old array
        unsigned int oldset = ((subset & ~(1 << i)) <<  1) | 1;
        // Go from i to e, from e to j in each possible way
        int around = part[set ^ subset][j] + last[oldset][i + 1] + transition[i][k - 1];

        if (around < bestCost) {
          bestCost  = around;
          *selected = set ^ subset;
          best      = i;
        }
      }
    }

    return best;
  }

  /**
   * Given a selection that does not contain e, or has e as last element, 
   * compute the last element and selection for the next iteration
   */
  int selectionNotContained(int **last, int **transition, unsigned int selection) {
    // e is the last element of this path
    int best = INFINITY;
    int j = 0;
    Bits endsI(selection);

    for (BitsetElementIterator it = endsI.begin(); it != endsI.end(); ++it) {
      int i = *it;
      unsigned int theset = (selection & ~(1 << i)) << 1 | 1;
      int around = last[theset][i + 1] + transition[i][k - 1];
      if (around < best) {
        best = around;
        j = i;
      }
    }

    return j;
  }


  /**
   * Part: for each set S and each element a not in S, the cost of the
   *       optimal path from e to a
   */ 
  void dynamic(int **last, int **compute, int **part, int **transition) {
    unsigned int small = 1 << (k - 1);

    // For each set S not containing e = k - 1
    for (unsigned int set = 0; set < small; ++set) {
      // S U {b} in last
      unsigned int theset = (set << 1) | 1;

      // Iterate over each element not in S: 
      Bits bits(~set & (small - 1));
      for (BitsetElementIterator it = bits.begin(); it != bits.end(); ++it) {
        // For each i as last index
        int i = *it;
        // CASE 1: x = e
        int around = last[theset][i + 1] + transition[i][k - 1];
        if (around < compute[set | (1 << i)][k - 1]) 
          compute[set | (1 << i)][k - 1] = around;

        // CASE 2: x != e, S does not contain x
        compute[set][i] = last[theset][i + 1];
      }
    }

    // CASE 3: S contains x
    for (unsigned int set = 0; set < small; ++set) {
      // For each set S containing x
      unsigned int theset = set | small;
      // For each element not in S: possible end nodes of the path
      Bits endsJ(~set & (small - 1));
      // For each subset S' of S
      //for (unsigned int subset = (set - 1) & set; subset; subset = (subset - 1) & set) {
      for (unsigned int subset = set; subset; subset = (subset - 1) & set) {

        // We examine, for each subset S', whether or not it is best to go from node 0
        // to k using S - S', and from k to j using S'.
        // To this end, we iterate over all elements i of S' and try them as end node. 
        Bits endsI(subset);
        for (BitsetElementIterator iti = endsI.begin(); iti != endsI.end(); ++iti) {
          for (BitsetElementIterator itj = endsJ.begin(); itj != endsJ.end(); ++itj) {
            int j = *itj;
            // For each i as last index
            int i = *iti; 
            // Index of subset in the old array
            unsigned int oldset = ((subset & ~(1 << i)) <<  1) | 1;
            // TODO: can oldset and preceed k - 1 ? can it preceed (set ^ subset) -> 
            // Compute a precedence vector for each subset of set!
            int firstCost  = last[oldset][i + 1] + transition[i][k - 1];
            int secondCost = part[set ^ subset][j];
            // Go from i to e, from e to j in each possible way
            int around = firstCost + secondCost;

            if (around < compute[theset][j]) {
              compute[theset][j] = around;
            }
          }
        }
      }
    }
  }


};

/**
 * Idea: better bounding if we know for each set S what is the minimal cost
 * over all end nodes j.
 */

/**
 * Add selection costs to transition matrix
 * Solve optimal -> make selection -> add cost to chosen edges -> solve -> select -> repeat
 */
#endif

