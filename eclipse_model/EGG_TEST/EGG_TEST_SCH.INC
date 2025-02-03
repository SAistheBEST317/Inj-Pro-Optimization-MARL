-- Generated [
-- Format      : ECLIPSE keywords (ASCII)
-- Exported by : Petrel 2016.4 (64-bit) Schlumberger
-- User name   : Administrator
-- Date        : Friday, November 25 2022 09:30:30
-- Project     : EGG.pet
-- Generated ]

RPTSCHED                               -- Generated : Petrel
  FIP WELLS /

RPTRST                                 -- Generated : Petrel
  BASIC=3 FLOWS FREQ /

SKIP                                   -- Generated : Petrel
--Hint: Drag producers into PROD FOLDER, injectors into INJ FOLDER, set target Field production rate in Group rate production control rule, check the production BHP limit in Well pressure production control rule and set injection BHP limit in Well water injection control rule.

ENDSKIP                                -- Generated : Petrel

WELSPECS                               -- Generated : Petrel
--'INJECT1' is the simulation well name used to describe flow from 'INJECT1'
--'INJECT3' is the simulation well name used to describe flow from 'INJECT3'
--'PROD1' is the simulation well name used to describe flow from 'PROD1'
--
  INJECT1 'GROUP 1' 5 34 1* LIQ /
  INJECT3 'GROUP 1' 2 12 1* LIQ /
  PROD1 'GROUP 1' 16 20 1* WATER /
  /

COMPDAT                                -- Generated : Petrel
  INJECT1 5 34 1 1 OPEN 1* 5.0581E+001 0.20000 2608.00 0.00 1* Z 1.58 /
  INJECT1 5 34 2 2 OPEN 1* 3.1822E+001 0.20000 1640.80 0.00 1* Z 1.58 /
  INJECT1 5 34 3 3 OPEN 1* 4.2396E+001 0.20000 2186.00 0.00 1* Z 1.58 /
  INJECT1 5 34 4 4 OPEN 1* 2.9968E+001 0.20000 1545.20 0.00 1* Z 1.58 /
  INJECT1 5 34 5 5 OPEN 1* 3.4910E+001 0.20000 1800.00 0.00 1* Z 1.58 /
  INJECT1 5 34 6 6 OPEN 1* 3.9743E+001 0.20000 2049.20 0.00 1* Z 1.58 /
  INJECT1 5 34 7 7 OPEN 1* 3.9184E+001 0.20000 2020.38 0.00 1* Z 1.58 /
  INJECT3 2 12 1 1 OPEN 1* 6.7330E+001 0.20000 3471.60 0.00 1* Z 1.58 /
  INJECT3 2 12 2 2 OPEN 1* 7.8284E+001 0.20000 4036.40 0.00 1* Z 1.58 /
  INJECT3 2 12 3 3 OPEN 1* 7.6833E+001 0.20000 3961.59 0.00 1* Z 1.58 /
  INJECT3 2 12 4 4 OPEN 1* 4.5655E+001 0.20000 2354.01 0.00 1* Z 1.58 /
  INJECT3 2 12 5 5 OPEN 1* 7.8284E+001 0.20000 4036.40 0.00 1* Z 1.58 /
  INJECT3 2 12 6 6 OPEN 1* 6.2659E+001 0.20000 3230.79 0.00 1* Z 1.58 /
  INJECT3 2 12 7 7 OPEN 1* 3.5096E+001 0.20000 1809.58 0.00 1* Z 1.58 /
  PROD1 16 20 1 1 OPEN 1* 5.5352E+001 0.20000 2853.99 0.00 1* Z 1.58 /
  PROD1 16 20 2 2 OPEN 1* 3.3638E+001 0.20000 1734.40 0.00 1* Z 1.58 /
  PROD1 16 20 3 3 OPEN 1* 4.1946E+001 0.20000 2162.80 0.00 1* Z 1.58 /
  PROD1 16 20 4 4 OPEN 1* 5.4048E+001 0.20000 2786.80 0.00 1* Z 1.58 /
  PROD1 16 20 5 5 OPEN 1* 4.9293E+001 0.20000 2541.60 0.00 1* Z 1.58 /
  PROD1 16 20 6 6 OPEN 1* 5.4444E+001 0.20000 2807.20 0.00 1* Z 1.58 /
  PROD1 16 20 7 7 OPEN 1* 4.3823E+001 0.20000 2259.58 0.00 1* Z 1.58 /
  /

GRUPTREE                               -- Generated : Petrel
  'GROUP 1' FIELD /
  /

WCONPROD                               -- Generated : Petrel
  INJECT1 1* LRAT 1* 1* 1* 50.00 /
  INJECT3 1* LRAT 1* 1* 1* 50.00 /
  /

WCONINJE                               -- Generated : Petrel
  PROD1 WATER 1* RATE 50.00 /
  /

DATES                                  -- Generated : Petrel
  1 FEB 2022 /
  /

