#ifndef GENERAL_H
#define GENERAL_H

#define BIGRND 0x7fffffff
#define PR(x) cout << #x << ": " << x << endl
#define WFP int z; cin >> z

typedef unsigned int uint;

// Generate a random number between 0.0 and 1.0
double rnd01() {
  return ((double) random() / (double) BIGRND);
}

// Generate a random number between -1.0 and +1.0
double nrnd01() {
  return ((rnd01() * 2.0) - 1.0);
}

#endif // GENERAL_H
