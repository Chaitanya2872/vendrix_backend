"""Bill of Entry test data: real OCR output, not idealised text.

Every string below is the *verbatim* recogniser output for one page of a real
ICEGATE Bill of Entry print (SEZ import, seven invoices, nineteen pages),
captured at 300 DPI. Nothing has been cleaned up.

That is the point. A fixture typed out from the PDF would test the parser
against text the parser will never see: on this document class OCR drops the
slash in the invoice marker, reads "INURG6" as "INUG6", loses a digit from
the AD code, and emits the grid in detection order rather than reading order.
A parser that passes against tidied text and fails against this one has not
been tested at all.

Known damage in these samples, all at high recogniser confidence:

  PAGE_ONE    "|7.2" for the 17.2 kg gross weight; the TOT ASS VAL reads
              5525826 against a true 5528626; the whole D.MANIFEST row
              (MAWB, HAWB and their dates) is missing, having been dropped
              by the detector at full-page scale.
  PAGE_TWO    header strip degraded ("INUG6", "356107", "426"); the AD code
              reads 658007 for 6580007. Part-II content is clean.
  PAGE_NINE   two goods lines, both arithmetically exact. The printed page
              number is 8, not 9 -- the scan contains a duplicated sheet.
"""

PAGE_ONE = """\nPort Code
BENo
BE DateBE Type
INURG6
3560107
04/09/2026
Z
kelb
_EC/BR
BH
IRPY
SPE
36ABBCS5682H1Z2/G
INDIAN
CUSTOMS
CB CODE
AAACZ6666LCH001
TYPE
ANI
_EMN
PORT
NoN
L
8
0
I PE
PKG
(S)
|7.2
E3
PART-I-BILLOFENTRYSUMMARY
金
1.BE STATUS 2.MODE 3.DEF BE 4.KACHA 5.SEC 48 6.REIMP
7.ADV BE
11.FIRST12.PROV/
TTUS
8.ASSESS 9.EXAM10.HSS
FIRST COPY
(Y/N/P)
CHECK
FINAL
SEZ
T
N
N
N
N
N
3
3.I
SN  TE
EEN
SNTE
SIN
Saegertown
16.PORT OF SHIPMENT
Saegertown
1.IMPORTERNAME & ADDRESS
SEZ UNIT DTLS: IECABBCS5682H SAFRAN AIRICRAFT ENGINES
ECANT
SAFRAN AIRCRAFT ENGINES HYDERABAD PRIVATE LIMITED
HYDERABAD PRIVATE LIMITED SAFRAN AIRCRAFTENGI
SAFRAN AIRCRAFT ENGINES HYDERABAD P
B.
PLOT NO 6{3} GMR AEROSPACE &amp; IN
2.CB NAME 5 WAY LOGISTICS SOLUTIONSINDIA PVT LTD
HYDERABAD
3.AEO
500108
4.UCR
SUMY
AD CODE
6580007
1.BCD
2ACD
3.SWS
4.NCCD5.ADD
6.CVD
7.IGST
8.G.CESS 18.TOT.ASS VAL
C.DUTY
552862.7
0
55286.3
1104620
0
9.SG
0.ED
.S
0
0
5525826
2ATAA
71769
1712769
.IGM NO
2.IGM DATE 3.INW DATE 4.GIGMNO 5.GIGMDT 6.MAWB NO 7.DATE
8.HAWB NO 9.DATE 10.PKG 11.GW
.FES
3135165
04/09/2026
04/09/2026
DETAILS
132
172
0
E. BOND DETAILS
1.B N
2.PORT 3.BOND CD 4.DEBT AMT 5.BG AMT
1.SR NO 2.CHALLAN NO
3.PAID ON
4.AMOUNT(Rs.)
2002269336
INURG6
SZ
1712769
 EN
ETILS
1.WBE NO.
2.DATE
 T
WDE
INVOICE DETAILS - SUMMARY#
1.S.NO
OJIN
3.INV. AMT
4.CUR
O.
WH
URG6Z008
1
26208771
382.08
USD
H. PROCESSING
1.EVENT
2.DATE
3.TIME
EXCHANGERATE
3
26205951
298.94
USD
or G1a
USD
Submission
04-SEP-26
14:33
INR=INR
ETILS
Assessment
04-SEP-26
14:34
4
26203001
15060
USD
1 USD=95.25INR
5
26186041
2638.74
USD
xon
9
6381
80.4
USD
0OC
Finalisation
7
0 9 SEP 2026
26208851
25260.86
USD
1.SNO2.LCL3.TRUCK
4.SEAL
5.CONTAINER NUMBER
FCL
-10134
OOC NO.
Safran Aircraft Engines Hyderabad Pvt. Ltd.
DOC DATE
16150
INWARD
OUTWARD
AP Name
CHOCKALINGAM SRINIVASAN
SI.No
09/09/26
APID
BAOPS2371B
Date
at
Time :
i e
s ilgnatures
I IRD
 IO
Date: 2026.09.04 14:36:17 IST
Reason: CUSTOMS
Location: INDIA
A: DEF - Deferred Payment, REIMP - Reimport, ADV - Advance, P - Prior, HSS - HighSeaSale; B : CB - Customs Broker, AEO - Authorized
GLOSSARY
Economic Operator, UCR - Unique Customs Reference; D : GIGM - Gateway IGM; G: WBE - WareHouse BE: I: OOC - Out of Charge, # Refer Par
V for full lis of Invoices J:Refer Part V for full list of Containers, AP- Authorised Person
Page
1 Of 19
"""

PAGE_TWO = """\nortde

INUG6
356107
426
Z
EC/BR
B
IRPY
INDIAN
CUSTOMS
SPE
6
CB CODE
AAACZ6666LCH001
TYPE
PORT:
INV
TEM
CONT
IP
SON
L
8
0
PKG
1
G.WT (KGS)
17.2
BE0040920261434
PART-II-INVOICE & VALUATION DETAILS (InvoIce 1/7)
INVICE
1.S.NO 2.INVOICE NO.& DT. 3.PURCHASE ORDER NO &DT
4.LC NO &DATE
5.CONTRACT NO & DATE
1
26208771
31-AUG-26
1.BUYER'S NAME & ADDRESS
SAFRAN AIRCRAFT ENGINES HYDERABAD PRIVATE LIMITED
2.SELLER'S NAME & ADDRESS
RAES
SAFRAN AIRCRAFT ENGINES HYDERABAD P
PLOT NO 6{3} GMR AEROSPACE &amp; IN
HYDERABAD
500108
3.SUPPLIERNAME&ADDRESS/CLIENTDETAILS
GREENLEAF
4.THIRD PARTY NAME & ADDRESS
18695 GREENLEAF DRIVE
SAEGERTOWN PA
UNITED STATES
16433
5.AEO

ON
 I NCE
658007
VALUATI
SESMS
C
808
VOD
4. US
OTH
5I5.
9.RELTD 10.SVB CH 11.SVB NO
12.DATE13LOA
ST.
EES
1.C&B
2.CoC
3.CoP
No
4.HND CHG5.G&S6.DOC.CH
7.COO
8.R&LF 9.OTH COST 10.LD/ULD11.WS
12.OTC
13.MISC CHARGE 14.ASS. VALUE
1.S NO.
2.CTH
3.DESCRIPTION
36393.12
4.UNIT PRICE
1
82090090
WG-4125A,XSYTIN-1,INSERT
5.QUANTITY 6.UQC
7.AMOUNT
11.940000
PRECISION
32.000000 NOS
382.08
GROUNDGROOVING.CERAMIC
CUTTING TOOL
EEAILS
E
         
LRY
  t
  场
Qc d
Verify using ICETRAK Mobile App (Google Play Store) for authenticatlon & latest version details from ICEGATE Enquiry
Page
2 Of 19
"""

PAGE_NINE = """\nPort Code
BENo
BE Date BE Type
INURG6
3560107
04/09/2026
Z
MeLes
/
B2
RPY
SPE
36ABBCS5682H1Z2/G
INDIANCUSTOMS
CB CODE
AAACZ6666LCH001
TYPE
INV
TEM
CONT
POT:
NoN
L
BILL OF ENTRY FOR SEZ IMPORT Z TYPE
8
0
PKG
1
G.WT (KGS)
17.2
BE0040920261434
PART-II-INVOICE&VALUATION DETAILS(InVOICe
7/7)
NVICE
1.S.NO 2.INVOICE NO. & DT. 3.PURCHASE ORDER NO & DT
4.LC NO&DATE
5.CONTRACT NO &DATE
7
26208851
31-AUG-26
1.BUYER'S NAME & ADDRESS
SAFRAN AIRCRAFT ENGINES HYDERABAD PRIVATE LIMITED
2.SELLER'S NAME & ADDRESS
RES
SAFRAN AIRCRAFT ENGINES HYDERABAD P
PLOT NO 6{3} GMR AEROSPACE &amp; IN
HYDERABAD
500108
3.SUPPLIERNAME&ADDRESS/CLIENTDETAILS
4.THIRD PARTY NAME & ADDRESS
GREENLEAF
18695 GREENLEAF DRIVE
SAEGERTOWN PA
UNITED STATES
16433
AN
5.AEO
6. AD CODE
6580007
I.INV VALUE2.FREIGHT 3.INSURANCE
4.HSS.5.LOADING 6.COMMN 7.PAY TERMS
8.VALUATION METHOD
C.
25260.86
OTH
14.Cur USD
15.Term CIF
9.RELTD 10.SVB CH 11.SVB NO
12.DATE 13LOA
No
Dcs
ES
1.C&B
2.CoC
3.CoP
4.HND CHG 5.G&S
6.DOC.CH
7.COO
8.R&LF
9.OTH COST 10.LD/ULD11.WS
12.0TC
13.MISC CHARGE 14.ASS. VALUE
2406096.92
1.SNO.
T
DO
UI
9
AN
1
82090090
WG-4187A,XSYTIN-1,INSERT
13.190000
580.000000 NOS
7650.20
PRECISION
GROUNDGROOVING.CERAMIC
CUTTING TOOL
2
82090090
RCGN-4VA,XSYTIN-1,INSERT-
13.620000
1293.000000 NOS
PRECISION GROUNDROUND-V-
17610.66
BOTTOM,CERAMIC CUTTING TOOL
EL
E
A: LC - Letter of Credit; B : AD - Authorized Dealer; C : HSS - High Sea Sale; D : C&B Commission & Brokerage, CoC - Cost of Container, CoP - Co
GLOSSARY
of Packing, HND CHG - Handling Charges, G&S - Goods and Service input cost, DOC CH - Document Charges, CoO - Country of Origin Certificate
R&LF - Royalty and Licence Fees, LD/ULD -Loading Unloading Charges, WS - Warranty Services, OTC - Other Costs, CTH - Customs Tariff Head,
UQC-Unit Quantity Code
Page
8 Of 19
Verify using ICETRAK Mobile App (Google Play Store) for authentication & latest version details from ICEGATE Enquiry
"""

# The three pages as one document, in the order they appear in the PDF. The
# page footers are what `parser.split_pages` separates on.
FULL_DOCUMENT = "\n".join((PAGE_ONE, PAGE_TWO, PAGE_NINE))

# Ground truth, read off the 300-DPI render by eye. Asserted against rather
# than restated in each test, so a corrected reading changes one line.
EXPECTED = {
    "be_number": "3560107",
    "port_code": "INURG6",
    "gstin": "36ABBCS5682H1Z2",
    "cb_code": "AAACZ6666LCH001",
    "iec": "ABBCS5682H",
    "total_duty": "1712769",
    "bcd": "552862.7",
    "sws": "55286.3",
    "igst": "1104620",
    "assessable_value": "5528626",
    "invoice_count": 7,
}
