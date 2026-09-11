// Dump the per-PMT vector<double> branches of a Wire-Cell T_BDTvars tree for a
// given (run, subrun, event).
//
// TTree::Scan cannot expand a vector<double> inline -- it prints a blank column,
// which is why the branch looked empty. This reads the vector properly.
//
//   root -l -b -q 'dump_wc_pmt_info.C("/path/official.root",15014,234,11701)'
//
//   # just see what PMT-ish branches exist and their types:
//   root -l -b -q 'dump_wc_pmt_info.C("/path/official.root",15014,234,11701,"T_BDTvars","PMT",true)'
//
//   # widen the branch filter (e.g. all optical/flash branches):
//   root -l -b -q 'dump_wc_pmt_info.C("/path/official.root",15014,234,11701,"T_BDTvars","")'
//
// For run 15014 / subrun 234 / event 11701 it also prints OUR per-PMT numbers
// beside the dump, so the comparison is immediate. The key question for that
// event: opdet 18/24/21/20 report ~0 PE in our opflash while the charge predicts
// 794 / 7151 / 2756 / 5326 PE. Does the Wire-Cell vector see light there?
//
// INDEXING -- read this before concluding anything. Our obs_pe is OPDET-indexed
// (larlite larutil::Geometry::OpDetFromOpChannel, which is what filled
// merged_sp). Wire-Cell PMT vectors are conventionally OPCHANNEL-ordered. The
// two orderings differ on ALL 32 entries, so the macro prints the comparison
// under BOTH and reports which correlates better. Do not assume.

#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TObjArray.h>
#include <TString.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>

// larlite larutil::Geometry::OpDetFromOpChannel : opch -> opdet
static const int kOpch2Opdet[32] = {
   3,  5,  1,  6,  0,  2,  4,  9, 11,  7, 12,  8, 10, 14, 17, 13,
  18, 15, 16, 21, 22, 19, 24, 20, 23, 26, 29, 30, 25, 31, 27, 28
};

// Our numbers for run 15014 / subrun 234 / event 11701, OPDET-indexed.
// obs = larlite opflash simpleFlashBeam fPEperOpDet ; pred = PhotonLib x charge.
static const double kOursObs[32] = {
  0.00, 0.00, 0.00, 0.97, 0.00, 0.84, 3.23, 9.46, 8.02, 12.80, 23.54, 28.73,
  48.92, 143.47, 232.37, 0.00, 394.70, 434.23, 0.00, 3666.93, 62.41, 0.00,
  4504.86, 3681.30, 0.00, 759.16, 401.04, 325.18, 358.64, 180.34, 139.08, 104.48
};
static const double kOursPred[32] = {
  0.65, 0.50, 1.00, 0.80, 1.50, 1.40, 3.00, 6.40, 9.30, 9.80, 17.40, 16.80,
  34.90, 103.60, 164.30, 171.60, 327.30, 300.50, 794.20, 2883.70, 5326.20,
  2755.70, 3996.60, 2939.70, 7150.60, 879.90, 327.40, 281.10, 149.70, 168.90,
  103.90, 99.30
};

static double corr(const double* a, const double* b, int n) {
  double ma = 0, mb = 0;
  for (int i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  double sab = 0, sa = 0, sb = 0;
  for (int i = 0; i < n; ++i) {
    sab += (a[i] - ma) * (b[i] - mb);
    sa  += (a[i] - ma) * (a[i] - ma);
    sb  += (b[i] - mb) * (b[i] - mb);
  }
  if (sa <= 0 || sb <= 0) return 0.0;
  return sab / std::sqrt(sa * sb);
}

void dump_wc_pmt_info(const char* fname,
                      int run_want = 15014, int subrun_want = 234,
                      int event_want = 11701,
                      const char* treename = "T_BDTvars",
                      const char* pattern = "PMT",
                      bool list_only = false) {
  TFile* f = TFile::Open(fname, "READ");
  if (!f || f->IsZombie()) { std::cout << "!!! cannot open " << fname << "\n"; return; }
  TTree* T = (TTree*)f->Get(treename);
  if (!T) { std::cout << "!!! no tree " << treename << " in file\n"; f->ls(); return; }
  std::cout << ">>> " << treename << " : " << T->GetEntries() << " entries\n";

  // ---- pass 1: find the entry (only the id branches enabled = fast) --------
  Int_t run = 0, subrun = 0, event = 0;
  T->SetBranchStatus("*", 0);
  T->SetBranchStatus("run", 1);
  T->SetBranchStatus("subrun", 1);
  T->SetBranchStatus("event", 1);
  T->SetBranchAddress("run", &run);
  T->SetBranchAddress("subrun", &subrun);
  T->SetBranchAddress("event", &event);
  Long64_t found = -1;
  for (Long64_t i = 0; i < T->GetEntries(); ++i) {
    T->GetEntry(i);
    if (run == run_want && subrun == subrun_want && event == event_want) { found = i; break; }
  }
  if (found < 0) {
    std::cout << "!!! (" << run_want << "," << subrun_want << "," << event_want
              << ") not found\n";
    return;
  }
  std::cout << ">>> (" << run_want << "," << subrun_want << "," << event_want
            << ") -> entry " << found << "\n";

  // ---- pass 2: hook up the matching vector branches ------------------------
  T->SetBranchStatus("*", 1);
  T->ResetBranchAddresses();
  TObjArray* brs = T->GetListOfBranches();
  std::map<TString, std::vector<double>*> vd;
  std::map<TString, std::vector<float>*>  vf;
  std::map<TString, std::vector<int>*>    vi;
  TString pat(pattern);
  pat.ToLower();

  std::cout << "\n== branches matching \"" << pattern << "\" ==\n";
  for (int i = 0; i < brs->GetEntries(); ++i) {
    TBranch* b = (TBranch*)brs->At(i);
    TString name(b->GetName());
    TString lname(name); lname.ToLower();
    if (pat.Length() && !lname.Contains(pat)) continue;
    TString cls(b->GetClassName());
    std::cout << "   " << std::left << std::setw(34) << name
              << " " << (cls.Length() ? cls.Data() : "(scalar)") << "\n";
    if (cls == "vector<double>") {
      vd[name] = nullptr; T->SetBranchAddress(name, &vd[name]);
    } else if (cls == "vector<float>") {
      vf[name] = nullptr; T->SetBranchAddress(name, &vf[name]);
    } else if (cls == "vector<int>") {
      vi[name] = nullptr; T->SetBranchAddress(name, &vi[name]);
    }
  }
  if (vd.empty() && vf.empty() && vi.empty()) {
    std::cout << "\n(no vector branches matched -- widen `pattern`, e.g. \"\" for all)\n";
    return;
  }
  T->GetEntry(found);

  std::cout << "\n== vector sizes ==\n";
  for (auto& kv : vd) std::cout << "   " << std::left << std::setw(34) << kv.first
        << " vector<double> size=" << (kv.second ? (int)kv.second->size() : -1) << "\n";
  for (auto& kv : vf) std::cout << "   " << std::left << std::setw(34) << kv.first
        << " vector<float>  size=" << (kv.second ? (int)kv.second->size() : -1) << "\n";
  for (auto& kv : vi) std::cout << "   " << std::left << std::setw(34) << kv.first
        << " vector<int>    size=" << (kv.second ? (int)kv.second->size() : -1) << "\n";
  if (list_only) return;

  // ---- dump every vector, in its own native ordering -----------------------
  std::cout << "\n== per-index dump (index = the vector's OWN ordering) ==\n";
  std::cout << std::right << std::setw(5) << "idx";
  for (auto& kv : vd) std::cout << " | " << std::setw(16) << kv.first(0, 16);
  for (auto& kv : vf) std::cout << " | " << std::setw(16) << kv.first(0, 16);
  std::cout << "\n";
  size_t nmax = 0;
  for (auto& kv : vd) if (kv.second) nmax = std::max(nmax, kv.second->size());
  for (auto& kv : vf) if (kv.second) nmax = std::max(nmax, kv.second->size());
  for (size_t k = 0; k < nmax; ++k) {
    std::cout << std::right << std::setw(5) << k;
    for (auto& kv : vd) {
      std::cout << " | " << std::setw(16) << std::fixed << std::setprecision(3);
      if (kv.second && k < kv.second->size()) std::cout << kv.second->at(k);
      else std::cout << "-";
    }
    for (auto& kv : vf) {
      std::cout << " | " << std::setw(16) << std::fixed << std::setprecision(3);
      if (kv.second && k < kv.second->size()) std::cout << kv.second->at(k);
      else std::cout << "-";
    }
    std::cout << "\n";
  }

  // ---- side-by-side against our numbers (only for the traced event) --------
  if (!(run_want == 15014 && subrun_want == 234 && event_want == 11701)) return;
  for (auto& kv : vd) {
    if (!kv.second || kv.second->size() != 32) continue;
    const std::vector<double>& v = *kv.second;
    // read the WC vector as OPDET-indexed vs as OPCHANNEL-indexed
    double as_od[32], as_ch[32];
    for (int od = 0; od < 32; ++od) as_od[od] = v[od];
    for (int ch = 0; ch < 32; ++ch) as_ch[kOpch2Opdet[ch]] = v[ch];
    double c_od = corr(as_od, kOursObs, 32);
    double c_ch = corr(as_ch, kOursObs, 32);
    bool ch_wins = (c_ch >= c_od);
    std::cout << "\n===== " << kv.first << " vs OUR per-PMT numbers =====\n";
    std::cout << "  corr with our obs_pe: as OPDET " << std::setprecision(3) << c_od
              << " | as OPCHANNEL " << c_ch << "  ->  looks "
              << (ch_wins ? "OPCHANNEL-indexed" : "OPDET-indexed") << "\n\n";
    const double* wc = ch_wins ? as_ch : as_od;   // now opdet-indexed either way
    std::cout << std::setw(5) << "opdet" << std::setw(6) << "opch"
              << std::setw(12) << "ours_obs" << std::setw(12) << "ours_pred"
              << std::setw(12) << "WC" << "   note\n";
    double to = 0, tw = 0;
    for (int od = 0; od < 32; ++od) {
      to += kOursObs[od]; tw += wc[od];
      TString note;
      if (od == 15) note = "our DEAD tube (opch 17)";
      else if (kOursObs[od] < 5.0 && kOursPred[od] > 300.0)
        note = (wc[od] > 50.0) ? "<== WE SEE 0, WC SEES LIGHT"
                               : "<== both see nothing";
      std::cout << std::fixed << std::setprecision(2)
                << std::setw(5) << od << std::setw(6) << [&]{
                     for (int c = 0; c < 32; ++c) if (kOpch2Opdet[c] == od) return c;
                     return -1; }()
                << std::setw(12) << kOursObs[od] << std::setw(12) << kOursPred[od]
                << std::setw(12) << wc[od] << "   " << note << "\n";
    }
    std::cout << std::setw(11) << "TOTAL" << std::setw(12) << to
              << std::setw(12) << 29028.0 << std::setw(12) << tw << "\n";
    std::cout << "\n  Our flash total is 15525 PE while the charge predicts 29028.\n"
              << "  The four tubes reading ~0 (opdet 18/20/21/24) carry 16198 PE of\n"
              << "  prediction and deliver 62 -- i.e. about HALF the event's light is\n"
              << "  missing from our flash. If WC has that light, the run3 optical\n"
              << "  stream we ingested dropped it.\n";
  }
}
