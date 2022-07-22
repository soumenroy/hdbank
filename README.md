# hdbank

# Hybrid template bank

Create a template bank for nonprecessing binary using hybrid-geometric random template placement method.
For details of how this code works see:
*  S. Roy, A. S. Sengupta, and N. Thakor, Phys. Rev. D95, 104045 (2017), [1702.06771](https://arxiv.org/abs/1702.06771)
*  S. Roy, A. S. Sengupta,  and P. Ajith, Phys. Rev. D99, 024048 (2019), [1711.08743](https://arxiv.org/abs/1711.08743)

## Source personal virtaul environments
One key component for generating the hybrid bank is metric over the dimensionless chirp time coordinate. To compute the metric, we modified in the LALSimulation code base, which is available in various [**clusters**](https://wiki.ligo.org/Computing/LDG/ClusterLogin). At this moment, the code will work only after the source my personal virtual environments. The details are following below.
   * **CIT cluster:** `conda activate /home/soumen.roy/.conda/envs/hdbank-env/`
   
       lal_version: `LAL: '7.1.6'`,  lalsimulation_version: `'3.1.1'`

## Clone hybrid bank directory
Clone: `git clone https://git.ligo.org/soumen.roy/hdbank.git`

Checkout python3 version: `git checkout hdbank_py3`

## Generating hybrid template bank using O4 design PSD.
Hybrid template bank for the O4 design psd was constructed by seeding two precomputed template banks. The seed banks were constructed for BNS and BBH search space. Please find the details of the search space [here](https://git.ligo.org/soumen.roy/HybridBankResults/-/tree/master/O4DesignPSD).


*  O4 design PSD file is available [here](https://git.ligo.org/soumen.roy/HybridBankResults/-/blob/master/O4DesignPSD/psd/aligo_O4low.txt) and details of other design PSDs are available [here](https://dcc.ligo.org/LIGO-T2000012).
*  Create a seed template bank for BNS systems.
```sh
pycbc_geom_aligned_bank  --min-match 0.97 --random-seed 1915 --pn-order threePointFivePN --f-low 27 --f-low-column alpha6 --f-upper 1000 --delta-f  0.01 --min-mass1 1.  --max-mass1 2. --min-mass2 1.  --max-mass2  2. --min-total-mass 2.  --max-total-mass 4. --max-bh-spin-mag 0.998  --max-ns-spin-mag 0.05 --ns-bh-boundary-mass 2. --asd-file /home/soumen.roy/aLIGOBanks/O4Bank/design_psd/aligo_O4low.txt  --split-bank-num 20 --output-file seed_bns.xml --intermediate-data-file intermediate.hdf --metadata-file metadata.xml --workflow-name bns_seed_bank --supplement-config-file pegasus.ini
```
where `pegasus.ini` file 
```
[pegasus_profile]
condor|accounting_group = ligo.dev.o4.cbc.bns_spin.pycbcoffline
condor|request_disk = 1024
```
 then submit the dax file:
```sh
pycbc_submit_dax  bank_gen.dax
```
*  Create a seed template bank for BBH systems.

```sh
export OPENBLAS_MAIN_FREE=1
export OPENBLAS_NUM_THREADS=1

python lalapps_cbc_hdbank.py --approximant SEOBNRv4_ROM \
                    --min-mass1 5.0 --max-mass1 499.0 \
                    --min-mass2 5.0 --max-mass2 250.0  \
                    --min-total-mass 10.0 --max-total-mass 500.0 \
                    --nsbh-boundary-mass 2.0 \
                    --min-eta 0.1875  --max-eta 0.25 \
                    --min-bh-spin -0.998 --max-bh-spin 0.998 \
                    --min-ns-spin -0.05 --max-ns-spin 0.05 \
                    --min-match 0.98 \
                    --fref 15.0 --flow 15.0 --fhigh 1024.0 --df 0.1 \
                    --optimize-A3star-lattice-orientation \
                    --optimize-flow 0.995 \
                    --starting-point 13.0 7.0 \
                    --enable-exact-match \
                    --number-threads 32 \
                    --asd-file aligo_O4low.txt \
                    --gps-start-time 1241560818  --gps-end-time 1241589618 \
                    --channel-name H1L1 \
                    --random-list-size 5000000 \
                    --template-bin-size 1000 \
                    --random-seed 6161 \
                    --neighborhood-size 2.0 \
                    --output-filename BBH_SEED_BANK.xml.gz \
                    --write-metric \
                    --verbose\
```

* Create the final bank by seeding the precomputed banks.

```sh
export OPENBLAS_MAIN_FREE=1
export OPENBLAS_NUM_THREADS=1

python lalapps_cbc_hdbank.py \
                    --approximant SEOBNRv4_ROM \
                    --min-mass1 1.0 --max-mass1 499.0 \
                    --min-mass2 1.0 --max-mass2 250.0  \
                    --min-total-mass 2.0 --max-total-mass 500.0 \
                    --nsbh-boundary-mass 2.0 \
                    --min-eta 0.010000079477919938  --max-eta 0.25 \
                    --min-bh-spin -0.998 --max-bh-spin 0.998 \
                    --min-ns-spin -0.05 --max-ns-spin 0.05 \
                    --min-match 0.965 \
                    --fref 15.0 --flow 15.0 --fhigh 1024.0 --df 0.1 \
                    --optimize-A3star-lattice-orientation \
                    --starting-point 10.0 3.0 \
                    --optimize-flow 0.995 \
                    --enable-exact-match \
                    --number-threads 32 \
                    --asd-file aligo_O4low.txt \
                    --bank-seed seed_bns.xml BBH_SEED_BANK.xml \
                    --random-list-size 50000000 \
                    --template-bin-size 1000 \
                    --random-seed 6161 \
                    --neighborhood-size 2.0 \
                    --output-filename O4_DESIGN_OPT_FLOW_HYBRID_BANK.xml.gz \
                    --write-metric \
                    --verbose \

```
