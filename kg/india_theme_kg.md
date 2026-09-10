# India Thematic Stock Knowledge Graph
### Theme A — Strait of Hormuz disruption | Theme B — India AI infrastructure buildout

**Version:** 1.0 · **Compiled:** 5 August 2026 · **Universe:** NSE/BSE-listed Indian equities (+ key unlisted nodes for graph completeness)

> **Not investment advice.** This is a research scaffold — a map of *causal exposure*, not a view on price, valuation, or whether any exposure is already discounted. A node appearing under "beneficiary" only means a transmission channel exists. Verify every linkage against company filings, segment disclosures and current order books before acting. I'm not a financial advisor.

---

## 0. How to read this file

The graph is expressed as **typed nodes** and **typed directed edges**.

```
Node:  ID | TYPE | Label | key attributes
Edge:  SOURCE_ID -[EDGE_TYPE {attributes}]-> TARGET_ID
```

**Order** (1st / 2nd / 3rd) is a property of a *path*, not of a company. It is defined here as **hop distance from the shock node**:

| Order | Definition | Test |
|---|---|---|
| **1st** | The shock changes a P&L line item *directly and mechanically* | Does the company buy/sell/ship the disrupted thing itself? |
| **2nd** | The shock reaches the company through **one intermediate node** — a raw material, a customer, or a supplier | Does its input price or its customer's capex move because of the shock? |
| **3rd** | The shock reaches the company through **a macro variable, a policy response, or a behavioural change** | Does it arrive via rates, currency, subsidy, demand substitution, or sentiment? |

Overlap is expected and is a *feature*: the highest-information nodes in this graph are the ones with edges into **both** themes (see §7).

---

## 1. Ontology

### 1.1 Node types

| Type | Prefix | Meaning |
|---|---|---|
| `SECTOR` | `S_` | GICS-ish / Nifty sector or thematic bucket |
| `SUBSECTOR` | `SS_` | Narrower grouping within a sector |
| `COMPANY` | `C_` | Listed Indian company (ticker in attributes) |
| `PRIVATE_CO` | `PC_` | Unlisted/foreign entity that is causally important |
| `RAW_MATERIAL` | `RM_` | Commodity or feedstock |
| `PRODUCT` | `P_` | Manufactured good or service sold |
| `ASSET` | `AS_` | Physical asset: refinery, terminal, campus, port |
| `GEOGRAPHY` | `G_` | Country, chokepoint, region, cluster |
| `MACRO_VAR` | `M_` | Brent, INR, CPI, repo, CAD, freight, insurance |
| `POLICY` | `POL_` | Scheme, tariff, subsidy, regulation |
| `EVENT` | `E_` | The shock itself, or a discrete catalyst |
| `TECHNOLOGY` | `T_` | Technology / architecture creating demand |
| `CONSTRAINT` | `CN_` | Physical or supply bottleneck that gates the theme |
| `CUSTOMER` | `CU_` | Demand-side buyer (hyperscaler, government, farmer) |
| `INVESTOR` | `IN_` | Capital provider whose allocation moves the theme |

### 1.2 Edge types

| Edge | Semantics | Example |
|---|---|---|
| `CONTAINS` | Sector → constituent | `S_OIL_GAS -> C_ONGC` |
| `PRODUCES` | Company → output | `C_ONGC -> RM_CRUDE` |
| `NEEDS` | Product/company → input | `P_PAINT -> RM_SOLVENT` |
| `SUPPLIES_TO` | Vendor → customer | `C_KRN -> C_BLUESTAR` |
| `CUSTOMER_OF` | Inverse of SUPPLIES_TO | |
| `IMPORTS_VIA` | Entity → chokepoint/route | `G_INDIA -> G_HORMUZ` |
| `TRANSITS` | Commodity → chokepoint | `RM_LPG -> G_HORMUZ` |
| `SOURCED_FROM` | Commodity → origin geography | `RM_LNG -> G_QATAR` |
| `PRICE_LINKED_TO` | Node price co-moves with | `RM_CARBON_BLACK -> M_BRENT` |
| `BENEFITS_FROM` | Positive exposure | `C_ONGC -> E_HORMUZ` |
| `HURT_BY` | Negative exposure | `C_IOC -> M_BRENT` |
| `SUBSTITUTES_FOR` | Demand shifts here when other is dear | `RM_COAL -> RM_LNG` |
| `HEDGES` | Internal natural offset | `C_RIL segment hedge` |
| `CAUSES` | Event → macro move | `E_HORMUZ -> M_BRENT` |
| `CONSTRAINS` | Bottleneck gating growth | `CN_CRGO -> P_TRANSFORMER` |
| `ENABLED_BY` | Policy tailwind | `S_DATACENTRE -> POL_DC_TAX` |
| `REGULATED_BY` | Price/subsidy control | `C_HPCL -> POL_LPG_SUBSIDY` |
| `OWNS` / `JV_WITH` / `SUBSIDIARY_OF` | Corporate structure | `C_ADANIENT -> PC_ADANICONNEX` |
| `INVESTS_IN` | Capital flow | `IN_BLACKSTONE -> PC_LUMINA` |
| `LOCATED_IN` | Asset → geography | `AS_JAMNAGAR_DC -> G_GUJARAT` |
| `COMPETES_WITH` | Rivalry | |
| `EXPOSED_TO_ORDERBOOK` | Revenue tied to a region's capex | `C_LT -> G_GCC` |

---

## 2. THEME A — Strait of Hormuz disruption

### 2.1 State of the world (as of Aug 2026)

This is not a hypothetical scenario. Grounding facts to anchor the graph:

- Conflict from late Feb 2026 led the IRGC to declare the Strait closed on 2 Mar 2026; transits fell >95%. A partial reopening followed in June, but status remained volatile into August 2026 with recurring restrictions. Treat the shock node as **live and oscillating**, not one-shot.
- Brent moved from ~$70–72 pre-war to a $118–130 peak (physical cargoes higher), and has since oscillated in the $80s. **The graph should be run in both directions** — escalation and de-escalation flip the sign on every edge.
- India-specific dependency (9M FY26): **~41% of crude, ~55% of LNG, ~88% of LPG imports** transit Hormuz. Jan 2026 crude share via Hormuz rose to ~53%. India imports ~60% of its LPG consumption → roughly half of all household cooking gas passes this one chokepoint.
- Non-energy cargo: **~70% of India's urea imports** come from Gulf states (Oman the largest single supplier), plus DAP, sulphur and ammonia.
- Second-order maritime economics: war-risk premia went from ~0.125% of hull value per transit to 2.5–5% at peak (~$5m per VLCC), tanker freight PG→Asia up several hundred percent, Cape routing adding ~10–14 days.
- Policy response nodes activated: excise duty cuts on motor fuels, a **Natural Gas Control Order** rationing supply, customs duty exemptions on petrochemicals, bilateral safe-passage talks with Iran for Indian-flagged vessels.

### 2.2 Core shock and macro nodes

```
E_HORMUZ      | EVENT      | Strait of Hormuz closure/disruption | recurring, 2026
G_HORMUZ      | GEOGRAPHY  | Strait of Hormuz | ~20% of seaborne oil, ~20% of LNG trade
G_GCC         | GEOGRAPHY  | Gulf producers (SA, UAE, Qatar, Kuwait, Iraq, Oman)
G_QATAR       | GEOGRAPHY  | Qatar | India's largest LNG supplier
G_OMAN        | GEOGRAPHY  | Oman | India's largest urea supplier
G_RUSSIA      | GEOGRAPHY  | Russia | discounted crude, alternate LPG/LNG route
G_CAPE        | GEOGRAPHY  | Cape of Good Hope reroute | +3,800nm, +10–14 days
M_BRENT       | MACRO_VAR  | Brent crude price
M_LNG_SPOT    | MACRO_VAR  | JKM / spot LNG price
M_LPG_SAUDI_CP| MACRO_VAR  | Saudi LPG contract price
M_GRM         | MACRO_VAR  | Singapore/complex gross refining margin
M_FREIGHT     | MACRO_VAR  | VLCC / product tanker freight (WS)
M_WARRISK     | MACRO_VAR  | Marine war-risk insurance premium
M_INR         | MACRO_VAR  | USD/INR
M_CPI         | MACRO_VAR  | India CPI inflation
M_CAD         | MACRO_VAR  | Current account deficit
M_REPO        | MACRO_VAR  | RBI policy rate / bond yields
M_UREA        | MACRO_VAR  | Global urea/DAP/sulphur price
M_SUBSIDY     | MACRO_VAR  | Fertiliser + LPG subsidy bill (fiscal)
```

**Macro edge chain (the spine of Theme A):**

```
E_HORMUZ -[CAUSES {sign:+}]-> M_BRENT
E_HORMUZ -[CAUSES {sign:+}]-> M_LNG_SPOT
E_HORMUZ -[CAUSES {sign:+, magnitude:high}]-> M_LPG_SAUDI_CP
E_HORMUZ -[CAUSES {sign:+, magnitude:extreme}]-> M_WARRISK
E_HORMUZ -[CAUSES {sign:+, magnitude:extreme}]-> M_FREIGHT
E_HORMUZ -[CAUSES {sign:+}]-> M_GRM            // product cracks widen on supply loss
E_HORMUZ -[CAUSES {sign:+}]-> M_UREA
M_BRENT  -[CAUSES {sign:+, elasticity:"$10/bbl ≈ -0.1 to -0.2pp GDP"}]-> M_CAD
M_CAD    -[CAUSES {sign:+}]-> M_INR            // depreciation
M_BRENT  -[CAUSES {sign:+}]-> M_CPI
M_INR    -[CAUSES {sign:+}]-> M_CPI            // imported inflation
M_CPI    -[CAUSES {sign:+}]-> M_REPO
M_UREA   -[CAUSES {sign:+}]-> M_SUBSIDY
M_LPG_SAUDI_CP -[CAUSES {sign:+}]-> M_SUBSIDY
RM_CRUDE -[TRANSITS {share:"~41–53% of India imports"}]-> G_HORMUZ
RM_LNG   -[TRANSITS {share:"~55%"}]-> G_HORMUZ
RM_LPG   -[TRANSITS {share:"~88–90%"}]-> G_HORMUZ
RM_UREA  -[TRANSITS {share:"~70% of imports"}]-> G_HORMUZ
RM_SULPHUR -[TRANSITS {share:"high"}]-> G_HORMUZ
RM_AMMONIA -[TRANSITS]-> G_HORMUZ
G_INDIA -[IMPORTS_VIA]-> G_HORMUZ
G_INDIA -[IMPORTS_VIA {role:"substitute route"}]-> G_RUSSIA
G_CAPE  -[SUBSTITUTES_FOR {cost:"+10–14 days"}]-> G_HORMUZ
```

---

### 2.3 FIRST ORDER — direct physical/price exposure

#### 2.3.1 Upstream producers — **positive**

```
C_ONGC       | COMPANY | Oil & Natural Gas Corp | ONGC
C_OIL        | COMPANY | Oil India | OIL
C_GAIL       | COMPANY | GAIL (India) | GAIL | gas transmission + petchem + LNG trading
C_SELAN      | COMPANY | Selan Exploration | small-cap crude producer
```
```
C_ONGC -[PRODUCES]-> RM_CRUDE
C_ONGC -[PRODUCES]-> RM_NATGAS
C_ONGC -[BENEFITS_FROM {order:1, channel:"realisation on domestic crude tracks Brent"}]-> M_BRENT
C_OIL  -[BENEFITS_FROM {order:1, channel:"same"}]-> M_BRENT
C_ONGC -[HURT_BY {order:1, channel:"windfall tax / SAED reimposition risk"}]-> POL_WINDFALL_TAX
C_ONGC -[HURT_BY {order:2, channel:"OVL assets and import-parity ceiling on APM gas"}]-> POL_APM_GAS
C_GAIL -[HEDGES {note:"gains on gas trading/transmission volume, loses on petchem feedstock + LNG sourcing cost"}]-> E_HORMUZ
C_ONGC -[OWNS]-> C_MRPL
C_ONGC -[OWNS {stake:"~51%"}]-> C_HPCL
```
> **Nuance:** the upstream "windfall" is politically fragile. A special additional excise duty (windfall tax) has historically clawed back the gain. `POL_WINDFALL_TAX` is the single most important edge to monitor on this branch.

#### 2.3.2 Refiners & oil marketing — **mixed to negative**

```
C_IOC     | COMPANY | Indian Oil | IOC     | refining + marketing + LPG
C_BPCL    | COMPANY | Bharat Petroleum | BPCL
C_HPCL    | COMPANY | Hindustan Petroleum | HINDPETRO | highest LPG mix of the three
C_MRPL    | COMPANY | Mangalore Refinery | MRPL | pure refiner, Gulf-heavy crude slate
C_CPCL    | COMPANY | Chennai Petroleum | CHENNPETRO | pure refiner
C_RIL     | COMPANY | Reliance Industries | RELIANCE | O2C + upstream KG-D6 + export refinery
PC_NAYARA | PRIVATE_CO | Nayara Energy | Rosneft-linked, Vadinar
```
```
C_IOC  -[NEEDS]-> RM_CRUDE
C_IOC  -[HURT_BY {order:1, channel:"LPG under-recovery — pump/cylinder price is administered"}]-> M_LPG_SAUDI_CP
C_HPCL -[HURT_BY {order:1, magnitude:high, channel:"largest LPG share of sales"}]-> M_LPG_SAUDI_CP
C_BPCL -[HURT_BY {order:1}]-> M_LPG_SAUDI_CP
C_IOC  -[HURT_BY {order:1, channel:"inventory loss on falling crude; marketing margin squeeze on rising crude"}]-> M_BRENT
C_IOC  -[BENEFITS_FROM {order:1, channel:"GRM expansion when product cracks widen"}]-> M_GRM
C_MRPL -[BENEFITS_FROM {order:1, channel:"pure refining, no marketing under-recovery"}]-> M_GRM
C_CPCL -[BENEFITS_FROM {order:1}]-> M_GRM
C_MRPL -[HURT_BY {order:1, channel:"crude slate concentrated in Gulf grades; substitution costly"}]-> E_HORMUZ
C_RIL  -[HEDGES {note:"O2C margin gains vs feedstock cost + KG-D6 gas realisation up; net long crude complexity"}]-> E_HORMUZ
C_RIL  -[BENEFITS_FROM {order:1, channel:"flexible slate, can arbitrage discounted Russian/Atlantic barrels"}]-> M_GRM
C_IOC  -[REGULATED_BY]-> POL_LPG_SUBSIDY
C_IOC  -[BENEFITS_FROM {order:3, channel:"government excise cuts absorb part of the shock"}]-> POL_EXCISE_CUT
```
> **Nuance:** OMCs are *not* simply "crude up = bad." The sequence matters — a sharp crude spike causes marketing-margin compression and LPG under-recovery (bad), but the same event widens diesel/ATF cracks (good), and a subsequent crude *fall* creates inventory losses (bad) before margins normalise (good). Model the branch as a **regime**, not a sign.

#### 2.3.3 Gas value chain — **negative on volume, mixed on price**

```
C_PETRONET | COMPANY | Petronet LNG | PETRONET | Dahej/Kochi regas, Qatar LT contract
C_GSPL     | COMPANY | Gujarat State Petronet | GSPL
C_IGL      | COMPANY | Indraprastha Gas | IGL   | CGD Delhi NCR
C_MGL      | COMPANY | Mahanagar Gas | MGL      | CGD Mumbai
C_GUJGAS   | COMPANY | Gujarat Gas | GUJGASLTD  | industrial CGD, Morbi ceramics cluster
C_ATGL     | COMPANY | Adani Total Gas | ATGL
C_GSFC     | COMPANY | Gujarat State Fertilizers & Chemicals | GSFC
```
```
C_PETRONET -[HURT_BY {order:1, channel:"regas throughput falls if Qatari cargoes are curtailed/force-majeured"}]-> E_HORMUZ
C_PETRONET -[HEDGES {note:"tolling model insulates margin per unit; volume is the exposed variable"}]-> M_LNG_SPOT
C_IGL -[HURT_BY {order:1, channel:"APM gas shortfall forces costly spot LNG in the mix"}]-> M_LNG_SPOT
C_MGL -[HURT_BY {order:1}]-> M_LNG_SPOT
C_GUJGAS -[HURT_BY {order:1, magnitude:high, channel:"industrial customers switch to propane/coal gasifier when LNG spikes → volume loss"}]-> M_LNG_SPOT
C_IGL -[BENEFITS_FROM {order:2, channel:"CNG vs petrol/diesel price differential widens → conversions rise"}]-> M_BRENT
C_GUJGAS -[REGULATED_BY]-> POL_GAS_CONTROL_ORDER
```
> **Nuance:** CGD is a two-sided node. Household/CNG volumes *benefit* from expensive petrol (substitution), while industrial volumes *lose* (customers switch away from gas). Gujarat Gas sits on the losing side of that split; IGL/MGL on the winning side.

#### 2.3.4 Shipping, ports & logistics — **strongly positive (tankers), negative (containers)**

```
C_GESHIP  | COMPANY | Great Eastern Shipping | GESHIP | crude+product tankers, offshore
C_SCI     | COMPANY | Shipping Corporation of India | SCI
C_SEAMEC  | COMPANY | Seamec | offshore support
C_ADANIPORTS | COMPANY | Adani Ports & SEZ | ADANIPORTS
C_JSWINFRA | COMPANY | JSW Infrastructure | JSWINFRA
C_CONCOR  | COMPANY | Container Corporation of India | CONCOR
C_ALLCARGO| COMPANY | Allcargo Logistics | ALLCARGO
C_TCI     | COMPANY | Transport Corp of India | TCI
```
```
C_GESHIP -[BENEFITS_FROM {order:1, magnitude:high, channel:"tanker day-rates spike on tonne-mile expansion + vessel scarcity"}]-> M_FREIGHT
C_SCI    -[BENEFITS_FROM {order:1}]-> M_FREIGHT
C_GESHIP -[HURT_BY {order:1, channel:"war-risk premia + bunker cost + crew risk on Gulf transits"}]-> M_WARRISK
C_GESHIP -[BENEFITS_FROM {order:2, channel:"Cape rerouting absorbs global tonnage → tightens supply everywhere"}]-> G_CAPE
C_ADANIPORTS -[HURT_BY {order:2, channel:"EXIM volume mix, Gulf trade lane share, transshipment disruption"}]-> E_HORMUZ
C_CONCOR -[HURT_BY {order:2, channel:"container freight inflation + volume softness"}]-> M_FREIGHT
C_TCI -[HURT_BY {order:2, channel:"diesel is the dominant opex line"}]-> M_BRENT
```

#### 2.3.5 Aviation — **strongly negative**

```
C_INDIGO  | COMPANY | InterGlobe Aviation | INDIGO | ATF ≈ 35–40% of opex
C_SPICEJET| COMPANY | SpiceJet | SPICEJET
PC_AIRINDIA | PRIVATE_CO | Air India (Tata) | Gulf route density
```
```
C_INDIGO -[NEEDS]-> P_ATF
P_ATF -[PRICE_LINKED_TO {beta:"near 1"}]-> M_BRENT
C_INDIGO -[HURT_BY {order:1, magnitude:high}]-> M_BRENT
C_INDIGO -[HURT_BY {order:2, channel:"USD-denominated lease rentals + maintenance reserves"}]-> M_INR
C_INDIGO -[HURT_BY {order:2, channel:"Gulf/Europe route re-routing around conflict airspace = longer block hours"}]-> E_HORMUZ
```
> **Nuance:** aviation is the cleanest "triple-hit" node in the graph — fuel, currency and airspace all move against it simultaneously. That is rare; most nodes have at least one offsetting edge.

#### 2.3.6 Direct beneficiaries by substitution

```
C_COALINDIA | COMPANY | Coal India | COALINDIA
C_NTPC      | COMPANY | NTPC | NTPC
C_BALRAMPUR | COMPANY | Balrampur Chini | BALRAMCHIN | ethanol
C_TRIVENI   | COMPANY | Triveni Engineering | TRIVENI
C_SHREERENUKA | COMPANY | Shree Renuka Sugars | RENUKA
C_PRAJ      | COMPANY | Praj Industries | PRAJIND | ethanol/CBG/SAF tech
C_WAAREE    | COMPANY | Waaree Energies | WAAREEENER
C_PREMIER   | COMPANY | Premier Energies | PREMIERENE
```
```
RM_COAL -[SUBSTITUTES_FOR]-> RM_LNG
C_COALINDIA -[BENEFITS_FROM {order:1, channel:"gas-to-coal switching in power and industry"}]-> M_LNG_SPOT
C_NTPC -[BENEFITS_FROM {order:2, channel:"coal-based PLF rises as gas-based generation becomes uneconomic"}]-> M_LNG_SPOT
P_ETHANOL -[SUBSTITUTES_FOR]-> P_PETROL
C_BALRAMPUR -[BENEFITS_FROM {order:2, channel:"higher crude improves ethanol blending economics and procurement price case"}]-> M_BRENT
C_PRAJ -[BENEFITS_FROM {order:3, channel:"energy-security capex cycle: ethanol, CBG, SAF"}]-> E_HORMUZ
C_WAAREE -[BENEFITS_FROM {order:3, channel:"policy acceleration of renewables as an import-substitution strategy"}]-> E_HORMUZ
```

---

### 2.4 SECOND ORDER — reached via one intermediate node (a raw material or a customer)

The mechanism: `E_HORMUZ → M_BRENT → RM_x → COMPANY`. These are input-cost stories, and the key modifier on every edge is **pricing power** (ability to pass through) and **inventory lag** (typically 1–2 quarters).

#### 2.4.1 Crude-derivative raw material nodes

```
RM_NAPHTHA | RAW_MATERIAL | Naphtha | cracker feedstock
RM_PROPYLENE | RAW_MATERIAL | Propylene
RM_BENZENE | RAW_MATERIAL | Benzene
RM_CBFS    | RAW_MATERIAL | Carbon black feedstock oil
RM_BASEOIL | RAW_MATERIAL | Base oil (Group I/II) | lubricants + transformer oil
RM_SYNRUBBER | RAW_MATERIAL | Synthetic rubber (SBR/PBR)
RM_TIO2    | RAW_MATERIAL | Titanium dioxide
RM_SOLVENT | RAW_MATERIAL | Solvents / mineral turpentine
RM_HDPE    | RAW_MATERIAL | HDPE / LLDPE / PP resin
RM_PVC     | RAW_MATERIAL | PVC resin
RM_PTA_MEG | RAW_MATERIAL | PTA / MEG (polyester chain)
RM_VAM     | RAW_MATERIAL | Vinyl acetate monomer
RM_LAB     | RAW_MATERIAL | Linear alkyl benzene (detergents)
RM_PETCOKE | RAW_MATERIAL | Petroleum coke
RM_BITUMEN | RAW_MATERIAL | Bitumen
RM_LLP     | RAW_MATERIAL | Liquid light paraffin (hair oil)
```
```
RM_NAPHTHA -[PRICE_LINKED_TO {beta:"0.8–1.0"}]-> M_BRENT
RM_CBFS    -[PRICE_LINKED_TO {beta:"high"}]-> M_BRENT
RM_SYNRUBBER -[PRICE_LINKED_TO]-> M_BRENT
RM_BASEOIL -[PRICE_LINKED_TO]-> M_BRENT
RM_PVC     -[PRICE_LINKED_TO {note:"also gated by Chinese carbide-route supply → weaker beta"}]-> M_BRENT
RM_HDPE    -[PRICE_LINKED_TO]-> RM_NAPHTHA
RM_PETCOKE -[PRICE_LINKED_TO]-> M_BRENT
RM_LLP     -[PRICE_LINKED_TO]-> M_BRENT
```

#### 2.4.2 Paints & adhesives — **negative** (crude derivatives ≈ 50–60% of RM basket)

```
C_ASIANPAINT | COMPANY | Asian Paints | ASIANPAINT
C_BERGER     | COMPANY | Berger Paints | BERGEPAINT
C_KANSAI     | COMPANY | Kansai Nerolac | KANSAINER
C_INDIGOPNT  | COMPANY | Indigo Paints | INDIGOPNTS
C_AKZO       | COMPANY | Akzo Nobel India | AKZOINDIA
C_PIDILITE   | COMPANY | Pidilite Industries | PIDILITIND
C_GRASIM     | COMPANY | Grasim (Birla Opus) | GRASIM
```
```
P_PAINT -[NEEDS]-> RM_SOLVENT
P_PAINT -[NEEDS]-> RM_TIO2
P_PAINT -[NEEDS]-> RM_MONOMER
C_ASIANPAINT -[HURT_BY {order:2, lag:"1 quarter", mitigant:"strong pricing power, but a price war with Birla Opus/Grasim caps pass-through"}]-> M_BRENT
C_BERGER -[HURT_BY {order:2}]-> M_BRENT
C_INDIGOPNT -[HURT_BY {order:2, magnitude:high, mitigant:"weakest pricing power of the group"}]-> M_BRENT
C_PIDILITE -[HURT_BY {order:2, channel:"VAM is the single largest input"}]-> RM_VAM
```
> **Nuance:** this branch's edge weight is currently *dampened by competition, not by crude*. The Grasim/Birla Opus entry means paint companies cannot pass through cost the way they did in 2018 or 2022. Same shock, weaker transmission.

#### 2.4.3 Tyres & carbon black — **negative** (but natural rubber offsets)

```
C_MRF     | COMPANY | MRF | MRF
C_APOLLOTYRE | COMPANY | Apollo Tyres | APOLLOTYRE
C_CEAT    | COMPANY | CEAT | CEATLTD
C_JKTYRE  | COMPANY | JK Tyre | JKTYRE
C_BALKRISHNA | COMPANY | Balkrishna Industries | BALKRISIND | OTR exporter
C_PCBL    | COMPANY | PCBL (Phillips Carbon Black) | PCBL
C_HIMADRI | COMPANY | Himadri Speciality Chemical | HSCL
```
```
P_TYRE -[NEEDS]-> RM_SYNRUBBER
P_TYRE -[NEEDS]-> RM_CARBONBLACK
P_TYRE -[NEEDS]-> RM_NATRUBBER
RM_CARBONBLACK -[NEEDS]-> RM_CBFS
C_PCBL -[HURT_BY {order:2, channel:"CBFS feedstock; partially passed through via formula pricing"}]-> M_BRENT
C_PCBL -[BENEFITS_FROM {order:3, channel:"co-generated power sold at higher merchant tariffs"}]-> M_LNG_SPOT
C_MRF -[HURT_BY {order:2, mitigant:"replacement-market pricing power; ~40% of RM is natural rubber, which is weather- not crude-driven"}]-> M_BRENT
C_BALKRISHNA -[HEDGES {note:"export revenue in USD/EUR benefits from INR depreciation, offsetting RM inflation"}]-> M_INR
```

#### 2.4.4 Plastics, pipes & packaging — **negative on cost, mixed on inventory**

```
C_SUPREME | COMPANY | Supreme Industries | SUPREMEIND
C_ASTRAL  | COMPANY | Astral | ASTRAL
C_FINOLEXIND | COMPANY | Finolex Industries | FINPIPE | backward-integrated into PVC
C_PRINCEPIPE | COMPANY | Prince Pipes | PRINCEPIPE
C_POLYPLEX | COMPANY | Polyplex | POLYPLEX | BOPET
C_COSMO   | COMPANY | Cosmo First | COSMOFIRST
C_UFLEX   | COMPANY | UFlex | UFLEX
C_EPL     | COMPANY | EPL Ltd | EPL | laminated tubes
C_TIMETECHNO | COMPANY | Time Technoplast | TIMETECHNO
```
```
P_PVC_PIPE -[NEEDS]-> RM_PVC
C_SUPREME -[HURT_BY {order:2, channel:"resin cost + destocking by dealers when prices are volatile"}]-> RM_PVC
C_FINOLEXIND -[HEDGES {note:"PVC resin manufacturing offsets pipe input cost"}]-> RM_PVC
C_POLYPLEX -[HURT_BY {order:2}]-> RM_PTA_MEG
C_EPL -[HURT_BY {order:2, channel:"polymer laminate cost, contractual lag with FMCG customers"}]-> RM_HDPE
```
> **Nuance:** for pipes, *volatility* hurts more than *level*. Rising resin prices actually create inventory gains and dealer restocking; the damage comes on the way down.

#### 2.4.5 Specialty & commodity chemicals — **negative**

```
C_SRF     | COMPANY | SRF | SRF
C_NAVINFLUOR | COMPANY | Navin Fluorine | NAVINFLUOR
C_GFL     | COMPANY | Gujarat Fluorochemicals | FLUOROCHEM
C_AARTIIND | COMPANY | Aarti Industries | AARTIIND
C_DEEPAKNTR | COMPANY | Deepak Nitrite | DEEPAKNTR | phenol/acetone ex-propylene
C_ATUL    | COMPANY | Atul | ATUL
C_VINATI  | COMPANY | Vinati Organics | VINATIORGA
C_TATACHEM | COMPANY | Tata Chemicals | TATACHEM | soda ash, energy-intensive
C_GHCL    | COMPANY | GHCL | GHCL
C_ALKYLAMINE | COMPANY | Alkyl Amines | ALKYLAMINE
```
```
C_DEEPAKNTR -[HURT_BY {order:2, channel:"benzene/propylene feedstock; competing against Chinese phenol imports limits pass-through"}]-> RM_BENZENE
C_AARTIIND -[HURT_BY {order:2}]-> RM_BENZENE
C_TATACHEM -[HURT_BY {order:2, channel:"soda ash is gas/energy intensive"}]-> M_LNG_SPOT
C_GHCL -[HURT_BY {order:2, channel:"lignite/coal and gas cost"}]-> M_LNG_SPOT
C_SRF -[HURT_BY {order:2, channel:"chloromethanes, packaging film feedstock"}]-> M_BRENT
```

#### 2.4.6 Fertilisers & agri-inputs — **highest non-energy Hormuz sensitivity in the market**

```
C_CHAMBAL | COMPANY | Chambal Fertilisers | CHAMBLFERT | urea, gas feedstock
C_COROMANDEL | COMPANY | Coromandel International | COROMANDEL | DAP/NPK, phos acid + sulphur
C_GNFC    | COMPANY | Gujarat Narmada Valley Fert & Chem | GNFC
C_RCF     | COMPANY | Rashtriya Chemicals & Fertilizers | RCF
C_NFL     | COMPANY | National Fertilizers | NFL
C_DEEPAKFERT | COMPANY | Deepak Fertilisers & Petrochemicals | DEEPAKFERT | ammonia, TAN
C_PARADEEP | COMPANY | Paradeep Phosphates | PARADEEP
C_FACT    | COMPANY | FACT | FACT
C_MADRASFERT | COMPANY | Madras Fertilizers | MADRASFERT
C_UPL     | COMPANY | UPL | UPL
C_PIIND   | COMPANY | PI Industries | PIIND
C_RALLIS  | COMPANY | Rallis India | RALLIS
```
```
P_UREA -[NEEDS]-> RM_NATGAS
P_DAP  -[NEEDS]-> RM_PHOSACID
P_DAP  -[NEEDS]-> RM_SULPHUR
P_DAP  -[NEEDS]-> RM_AMMONIA
RM_SULPHUR -[SOURCED_FROM {share:"Saudi/UAE dominant"}]-> G_GCC
RM_UREA -[SOURCED_FROM {share:"Oman ~26 LMT, largest single supplier"}]-> G_OMAN
C_CHAMBAL -[HURT_BY {order:1, channel:"gas feedstock cost; partially neutralised by cost-plus urea subsidy but working capital balloons"}]-> M_LNG_SPOT
C_COROMANDEL -[HURT_BY {order:1, magnitude:high, channel:"sulphur + phos acid + ammonia all transit or originate in the Gulf"}]-> E_HORMUZ
C_PARADEEP -[HURT_BY {order:1}]-> RM_SULPHUR
C_DEEPAKFERT -[HEDGES {note:"own ammonia plant becomes a strategic asset when imported ammonia is scarce"}]-> RM_AMMONIA
C_CHAMBAL -[REGULATED_BY]-> POL_NBS_SUBSIDY
C_COROMANDEL -[HURT_BY {order:3, channel:"subsidy disbursal delay when the fiscal bill balloons → receivable stretch"}]-> M_SUBSIDY
```
> **Nuance:** Indian fertiliser P&Ls are subsidy-buffered, so the earnings hit is smaller than the commodity move — but the **balance sheet** hit (receivables, working capital, interest cost) is real and is the thing to model. Also note the *policy response* already observed: reroutes via Red Sea from Russia, Morocco, Jordan and record domestic urea production, which blunts the volume risk while preserving the cost risk.

#### 2.4.7 Cement, metals & energy-intensive manufacturing — **negative**

```
C_ULTRACEMCO | COMPANY | UltraTech Cement | ULTRACEMCO
C_AMBUJA  | COMPANY | Ambuja Cements | AMBUJACEM
C_SHREECEM | COMPANY | Shree Cement | SHREECEM
C_DALMIA  | COMPANY | Dalmia Bharat | DALBHARAT
C_HINDALCO | COMPANY | Hindalco | HINDALCO
C_NALCO   | COMPANY | National Aluminium | NATIONALUM
C_VEDL    | COMPANY | Vedanta | VEDL
C_JSWSTEEL | COMPANY | JSW Steel | JSWSTEEL
C_TATASTEEL | COMPANY | Tata Steel | TATASTEEL
```
```
P_CEMENT -[NEEDS]-> RM_PETCOKE
P_CEMENT -[NEEDS {note:"~25% of cost is freight"}]-> P_DIESEL
C_ULTRACEMCO -[HURT_BY {order:2, channel:"pet coke + diesel freight; ~15–20% of cost base"}]-> M_BRENT
C_HINDALCO -[HURT_BY {order:2, channel:"calcined petroleum coke, caustic soda, coal — power is ~35–40% of aluminium cost"}]-> M_BRENT
C_NALCO -[HEDGES {note:"alumina/aluminium LME price also rises in a commodity-wide risk-on shock"}]-> M_BRENT
```

#### 2.4.8 Consumer & FMCG — **negative, but shallow**

```
C_HUL     | COMPANY | Hindustan Unilever | HINDUNILVR
C_MARICO  | COMPANY | Marico | MARICO | LLP + copra
C_GODREJCP | COMPANY | Godrej Consumer | GODREJCP
C_DABUR   | COMPANY | Dabur India | DABUR
C_JYOTHYLAB | COMPANY | Jyothy Labs | JYOTHYLAB
C_CASTROL | COMPANY | Castrol India | CASTROLIND
C_GULFOIL | COMPANY | Gulf Oil Lubricants | GULFOILLUB
```
```
C_MARICO -[HURT_BY {order:2, channel:"liquid paraffin (crude derivative) in hair oils + HDPE packaging"}]-> RM_LLP
C_HUL -[HURT_BY {order:2, channel:"LAB for detergents, packaging polymer, palm-oil freight"}]-> M_BRENT
C_CASTROL -[HURT_BY {order:2, magnitude:high, channel:"base oil is ~50% of COGS and is Gulf-sourced"}]-> RM_BASEOIL
C_GULFOIL -[HURT_BY {order:2}]-> RM_BASEOIL
```

#### 2.4.9 Autos — **negative via demand, positive via EV mix shift**

```
C_MARUTI  | COMPANY | Maruti Suzuki | MARUTI
C_HEROMOTOCO | COMPANY | Hero MotoCorp | HEROMOTOCO
C_BAJAJAUTO | COMPANY | Bajaj Auto | BAJAJ-AUTO
C_TVSMOTOR | COMPANY | TVS Motor | TVSMOTOR
C_TATAMOTORS | COMPANY | Tata Motors | TATAMOTORS
C_MM      | COMPANY | Mahindra & Mahindra | M&M
C_OLECTRA | COMPANY | Olectra Greentech | OLECTRA
C_AMARAJA | COMPANY | Amara Raja Energy | ARE&M
C_EXIDE   | COMPANY | Exide Industries | EXIDEIND
```
```
C_HEROMOTOCO -[HURT_BY {order:3, channel:"petrol price → rural running cost → 2W demand elasticity"}]-> M_BRENT
C_MARUTI -[HURT_BY {order:2, channel:"steel, plastics, freight; entry-level demand most fuel-price elastic"}]-> M_BRENT
C_TATAMOTORS -[BENEFITS_FROM {order:3, channel:"EV total-cost-of-ownership advantage widens; EV registrations rose ~50% YoY Mar-2026"}]-> M_BRENT
C_OLECTRA -[BENEFITS_FROM {order:3, channel:"diesel-bus to e-bus conversion economics"}]-> M_BRENT
C_MM -[HURT_BY {order:3, channel:"tractor demand tracks farm profitability, which fertiliser inflation compresses"}]-> M_UREA
```

---

### 2.5 THIRD ORDER — via macro, policy, currency and behaviour

#### 2.5.1 Rates & credit channel — **negative**

```
C_BAJFINANCE | COMPANY | Bajaj Finance | BAJFINANCE
C_CHOLAFIN | COMPANY | Cholamandalam Investment | CHOLAFIN
C_SHRIRAMFIN | COMPANY | Shriram Finance | SHRIRAMFIN
C_MUTHOOT | COMPANY | Muthoot Finance | MUTHOOTFIN
C_MANAPPURAM | COMPANY | Manappuram Finance | MANAPPURAM
C_HDFCBANK | COMPANY | HDFC Bank | HDFCBANK
C_SBIN    | COMPANY | State Bank of India | SBIN
C_AUBANK  | COMPANY | AU Small Finance Bank | AUBANK
```
```
M_REPO -[CAUSES {sign:+}]-> M_COF   // NBFC cost of funds
C_BAJFINANCE -[HURT_BY {order:3, channel:"cost of funds repricing faster than asset yields"}]-> M_REPO
C_SHRIRAMFIN -[HURT_BY {order:3, channel:"CV borrower cash flows compress when diesel rises → asset quality"}]-> M_BRENT
C_CHOLAFIN -[HURT_BY {order:3, channel:"vehicle finance book, same mechanism"}]-> M_BRENT
C_MUTHOOT -[BENEFITS_FROM {order:3, channel:"geopolitical risk → gold rally → higher LTV headroom and AUM growth"}]-> E_HORMUZ
C_MANAPPURAM -[BENEFITS_FROM {order:3}]-> E_HORMUZ
```

#### 2.5.2 Currency channel — **positive for exporters**

```
C_TCS     | COMPANY | Tata Consultancy Services | TCS
C_INFY    | COMPANY | Infosys | INFY
C_HCLTECH | COMPANY | HCL Technologies | HCLTECH
C_WIPRO   | COMPANY | Wipro | WIPRO
C_LTIM    | COMPANY | LTIMindtree | LTIM
C_SUNPHARMA | COMPANY | Sun Pharmaceutical | SUNPHARMA
C_DRREDDY | COMPANY | Dr Reddy's Laboratories | DRREDDY
C_CIPLA   | COMPANY | Cipla | CIPLA
C_DIVIS   | COMPANY | Divi's Laboratories | DIVISLAB
```
```
C_TCS -[BENEFITS_FROM {order:3, channel:"~1% INR depreciation ≈ 20–25bps EBIT margin"}]-> M_INR
C_SUNPHARMA -[BENEFITS_FROM {order:3}]-> M_INR
C_SUNPHARMA -[HURT_BY {order:2, channel:"solvents and KSM are crude-linked; but China, not the Gulf, is the sourcing risk"}]-> M_BRENT
C_DIVIS -[HURT_BY {order:2, channel:"solvent + energy cost in API manufacture"}]-> M_BRENT
```
> **Nuance:** the INR-benefit edge is real but small and usually swamped by demand-side news. Do not treat IT as a Hormuz "play" — treat it as a partial portfolio hedge.

#### 2.5.3 Gulf remittance & NRI-deposit channel — **the most under-mapped 3rd-order branch**

```
C_FEDERALBNK | COMPANY | Federal Bank | FEDERALBNK | highest NRI deposit share among Indian banks
C_SIB     | COMPANY | South Indian Bank | SOUTHBANK
C_CSBBANK | COMPANY | CSB Bank | CSBBANK
C_DHANBANK | COMPANY | Dhanlaxmi Bank | DHANBANK
C_MUTHOOTFIN | COMPANY | (see gold financiers above)
G_KERALA  | GEOGRAPHY | Kerala | remittance-dependent state economy
```
```
G_KERALA -[EXPOSED_TO_ORDERBOOK {channel:"Gulf expatriate employment"}]-> G_GCC
C_FEDERALBNK -[HEDGES {note:"INR depreciation lifts NRI remittance inflows and deposit accretion (+), but a Gulf construction slowdown from prolonged conflict cuts expat employment (−)"}]-> E_HORMUZ
C_FEDERALBNK -[BENEFITS_FROM {order:3, channel:"NRE/FCNR deposit inflows spike when INR weakens"}]-> M_INR
C_SIB -[BENEFITS_FROM {order:3}]-> M_INR
C_CSBBANK -[HURT_BY {order:3, channel:"gold loan LTV risk if gold reverses post-crisis"}]-> E_HORMUZ
```
> This branch matters disproportionately for Kerala- and Gujarat-anchored financials and consumer names, and it is almost never modelled in sell-side Hormuz notes.

#### 2.5.4 Gulf order-book exposure — **negative** (capex deferral risk)

```
C_LT      | COMPANY | Larsen & Toubro | LT | large Middle East hydrocarbon/infra order book
C_KEC     | COMPANY | KEC International | KEC
C_KALPATPOWR | COMPANY | Kalpataru Projects | KPIL
C_AFCONS  | COMPANY | Afcons Infrastructure | AFCONS
C_VOLTAS  | COMPANY | Voltas | VOLTAS | Gulf electro-mechanical projects legacy
```
```
C_LT -[EXPOSED_TO_ORDERBOOK {share:"material share of order inflow from Saudi/UAE/Qatar"}]-> G_GCC
C_LT -[HEDGES {note:"high oil price funds GCC sovereign capex (+) but active conflict defers award timelines and raises execution risk (−)"}]-> E_HORMUZ
C_KEC -[EXPOSED_TO_ORDERBOOK]-> G_GCC
C_VOLTAS -[EXPOSED_TO_ORDERBOOK {segment:"international projects, legacy Gulf MEP"}]-> G_GCC
```
> **Nuance:** this edge flips sign with duration. A *sustained high oil price without conflict* is the best state for L&T's Gulf book. *Conflict itself* is the bad state. The two are correlated but not identical — that distinction is where the alpha is.

#### 2.5.5 Defence, security & risk-premium beneficiaries

```
C_HAL     | COMPANY | Hindustan Aeronautics | HAL
C_BEL     | COMPANY | Bharat Electronics | BEL
C_BDL     | COMPANY | Bharat Dynamics | BDL
C_SOLARINDS | COMPANY | Solar Industries | SOLARINDS
C_COCHINSHIP | COMPANY | Cochin Shipyard | COCHINSHIP
C_MAZDOCK | COMPANY | Mazagon Dock | MAZDOCK
C_DATAPATTNS | COMPANY | Data Patterns | DATAPATTNS
C_ASTRAMICRO | COMPANY | Astra Microwave | ASTRAMICRO
C_GRSE    | COMPANY | Garden Reach Shipbuilders | GRSE
```
```
C_BEL -[BENEFITS_FROM {order:3, channel:"defence budget re-prioritisation + naval escort/surveillance demand"}]-> E_HORMUZ
C_COCHINSHIP -[BENEFITS_FROM {order:3, channel:"naval order flow + ship repair demand from rerouted tonnage"}]-> E_HORMUZ
C_SOLARINDS -[BENEFITS_FROM {order:3, channel:"defence explosives/ammunition demand"}]-> E_HORMUZ
```

#### 2.5.6 Insurance & risk transfer

```
C_GICRE   | COMPANY | GIC Re | GICRE
C_NIACL   | COMPANY | New India Assurance | NIACL
C_ICICIGI | COMPANY | ICICI Lombard | ICICIGI
```
```
C_GICRE -[HEDGES {note:"marine/war-risk premium rates harden (+) but claims frequency and retrocession cost rise (−)"}]-> M_WARRISK
C_NIACL -[BENEFITS_FROM {order:3, channel:"marine hull and cargo rate hardening"}]-> M_WARRISK
```

---

## 3. THEME B — India AI infrastructure buildout

### 3.1 State of the world (as of Aug 2026)

- Operational capacity: ~1.8 GW IT load (H1 2026 additions of 258 MW, +59% YoY), from ~375 MW in 2020. Projections: >7 GW by 2030 (Savills), ~12 GW by 2030 (Wood Mackenzie, ~40% CAGR).
- Ministry of Power now projects **26.3 GW of incremental AI data-centre load by FY32** — roughly double its own March 2026 estimate of 13.56 GW. This is the single most important number in Theme B, because it is the constraint that converts a compute story into a **power, transmission and grid-stability story**.
- Announced private capital: Reliance ~$110bn and Adani ~$100bn (AdaniConneX scaling 2 GW → 5 GW by 2035); Google's ~$15bn Visakhapatnam AI hub; Meta leasing a 168 MW AI-ready facility from Reliance at Jamnagar; OpenAI–Tata starting at 100 MW; Microsoft, AWS, Blackstone (Neysa, AirTrunk) all active.
- Policy: Budget 2026-27 introduced a **21-year tax holiday to March 2047** for foreign companies providing global cloud services from Indian "Specified Data Centers" notified by MeitY, with Indian-customer services routed through local reseller entities. Plus IndiaAI Mission compute procurement and IT-hardware PLI.
- Physical shape of demand: ~75% of the pipeline is shifting to **liquid cooling**, rack densities heading toward 1 MW; power is ~50% of operating cost; water consumption ~150bn litres/year heading to ~358bn by 2030, over half in water-stressed regions.
- Clusters: Mumbai, Chennai, Hyderabad, Delhi NCR, Bengaluru, Pune, Noida ≈ 65% of facilities; new anchors at Visakhapatnam and Jamnagar.

### 3.2 Core demand and constraint nodes

```
E_AIBUILD   | EVENT | India AI infrastructure capex cycle | 2024–2032
T_GENAI     | TECHNOLOGY | Generative AI / LLM training & inference
T_LIQUIDCOOL| TECHNOLOGY | Direct-to-chip / immersion liquid cooling
T_HIGHDENSITY | TECHNOLOGY | High-density racks (50–1000 kW)
CU_HYPERSCALER | CUSTOMER | Google, Meta, Microsoft, AWS, OpenAI
CU_INDIAAI  | CUSTOMER | IndiaAI Mission (sovereign compute procurement)
CU_GCC_ENT  | CUSTOMER | Global Capability Centres / Indian enterprises
POL_DC_TAX  | POLICY | 21-year tax holiday for global cloud from India (to 2047)
POL_INDIAAI | POLICY | IndiaAI Mission
POL_PLI_IT  | POLICY | IT hardware PLI (3–4% on incremental sales)
POL_SEMICON | POLICY | India Semiconductor Mission / OSAT incentives
CN_POWER    | CONSTRAINT | Grid power availability + 26.3 GW incremental load by FY32
CN_TRANSMISSION | CONSTRAINT | ISTS/InSTS evacuation capacity
CN_GRIDSTAB | CONSTRAINT | Load ramp volatility → STATCOM/BESS/synchronous condenser need
CN_WATER    | CONSTRAINT | Water availability in stressed clusters
CN_GPU      | CONSTRAINT | GPU import dependency, 36–52 week lead times
CN_HBM      | CONSTRAINT | HBM memory sold out; CoWoS advanced packaging shortage
CN_CRGO     | CONSTRAINT | CRGO electrical steel — imported, gates transformer output
CN_LAND     | CONSTRAINT | Infra-ready land with power + fibre + water
CN_SKILLS   | CONSTRAINT | Liquid-cooling / HV commissioning talent
```
```
T_GENAI -[CAUSES]-> E_AIBUILD
CU_HYPERSCALER -[CAUSES {via:"lease + build commitments"}]-> E_AIBUILD
POL_DC_TAX -[ENABLES]-> E_AIBUILD
E_AIBUILD -[CONSTRAINED_BY {severity:critical}]-> CN_POWER
E_AIBUILD -[CONSTRAINED_BY {severity:critical}]-> CN_TRANSMISSION
E_AIBUILD -[CONSTRAINED_BY {severity:high}]-> CN_GPU
E_AIBUILD -[CONSTRAINED_BY {severity:high}]-> CN_WATER
T_HIGHDENSITY -[CAUSES]-> T_LIQUIDCOOL
CN_CRGO -[CONSTRAINS]-> P_TRANSFORMER
CN_GRIDSTAB -[CAUSES {demand_for:"BESS, STATCOM, synchronous condensers"}]-> P_GRIDEQUIP
```
> **Key structural insight for traversal:** in India the binding constraint is *not* capital and *not* land — it is **firm power delivered to a specific point on the map**. Every high-conviction node in Theme B should be tested against the question "does this ease the power/thermal/grid constraint?" Nodes that only ride the narrative fail that test.

---

### 3.3 FIRST ORDER — build, own, operate, and supply the core stack

#### 3.3.1 Operators & developers

```
C_ADANIENT | COMPANY | Adani Enterprises | ADANIENT | AdaniConneX JV (with EdgeConneX), 2→5 GW plan
C_RIL      | COMPANY | Reliance Industries | RELIANCE | Jamnagar AI-ready campus; Meta 168 MW lease
C_BHARTIARTL | COMPANY | Bharti Airtel | BHARTIARTL | Nxtra Data
C_TATACOMM | COMPANY | Tata Communications | TATACOMM | DC + subsea + fibre
C_ANANTRAJ | COMPANY | Anant Raj | ANANTRAJ | DC conversion of legacy IT land
C_LODHA    | COMPANY | Macrotech Developers | LODHA | DC land/park development
C_SIFY     | COMPANY | Sify Technologies | (NASDAQ: SIFY) | DC + network
C_RAILTEL  | COMPANY | RailTel Corporation | RAILTEL | edge DC + govt cloud
PC_CTRLS   | PRIVATE_CO | CtrlS Datacenters | 2 GW renewable MoU with NTPC Green
PC_YOTTA   | PRIVATE_CO | Yotta (Hiranandani) | large GPU cloud deployments
PC_NXTRA   | PRIVATE_CO | Nxtra by Airtel
PC_STT     | PRIVATE_CO | STT GDC India (Tata Comms JV)
PC_EQUINIX | PRIVATE_CO | Equinix | first AI-ready DC in Chennai
PC_AIRTRUNK| PRIVATE_CO | AirTrunk (Blackstone-backed)
PC_NEYSA   | PRIVATE_CO | Neysa | Blackstone-backed GPU cloud
IN_BLACKSTONE | INVESTOR | Blackstone
IN_BROOKFIELD | INVESTOR | Brookfield
IN_GIC     | INVESTOR | GIC / sovereign wealth pools
AS_JAMNAGAR_DC | ASSET | Reliance Jamnagar AI campus | Gujarat
AS_VIZAG_DC | ASSET | Google AI hub, Visakhapatnam | Andhra Pradesh
```
```
C_ADANIENT -[JV_WITH]-> PC_ADANICONNEX
C_ADANIENT -[BENEFITS_FROM {order:1, magnitude:high}]-> E_AIBUILD
C_RIL -[OWNS]-> AS_JAMNAGAR_DC
CU_HYPERSCALER -[CUSTOMER_OF {deal:"Meta 168 MW lease"}]-> C_RIL
C_BHARTIARTL -[OWNS]-> PC_NXTRA
C_TATACOMM -[BENEFITS_FROM {order:1}]-> E_AIBUILD
C_ANANTRAJ -[BENEFITS_FROM {order:1, channel:"land bank → DC capacity conversion"}]-> E_AIBUILD
C_LODHA -[BENEFITS_FROM {order:1, channel:"industrial/DC land monetisation"}]-> CN_LAND
IN_BLACKSTONE -[INVESTS_IN]-> PC_NEYSA
IN_BLACKSTONE -[INVESTS_IN]-> PC_AIRTRUNK
PC_CTRLS -[JV_WITH {scale:"~2 GW renewable"}]-> C_NTPCGREEN
AS_VIZAG_DC -[LOCATED_IN]-> G_ANDHRA
```
> **Structural caveat:** there is **no clean listed pure-play DC operator in India**. The largest operators (CtrlS, Yotta, Nxtra, STT, AirTrunk, Equinix) are unlisted or foreign-listed. Listed exposure is therefore either *conglomerate-diluted* (Adani, Reliance, Airtel) or *picks-and-shovels*. This is the defining feature of Theme B and the reason the supply chain, not the operator, is where the listed money congregates.

#### 3.3.2 Electrical backbone — transformers, switchgear, UPS, gensets

```
C_ABB      | COMPANY | ABB India | ABB | switchgear, motors, drives, automation
C_SIEMENS  | COMPANY | Siemens India | SIEMENS
C_SIEMENSENER | COMPANY | Siemens Energy India | SIEMENSENER
C_HITACHIENER | COMPANY | Hitachi Energy India | POWERINDIA | HV transformers, grid
C_GEVERNOVA | COMPANY | GE Vernova T&D India | GVT&D
C_SCHNEIDER | COMPANY | Schneider Electric Infrastructure | SCHNEIDER
C_CGPOWER  | COMPANY | CG Power & Industrial Solutions | CGPOWER | + OSAT JV
C_VOLTAMP  | COMPANY | Voltamp Transformers | VOLTAMP | order book ~Rs 1,600–1,800cr
C_TRIL     | COMPANY | Transformers & Rectifiers India | TARIL
C_SHILCHAR | COMPANY | Shilchar Technologies | SHILCTECH
C_CUMMINSIND | COMPANY | Cummins India | CUMMINSIND | HHP gensets for DC backup
C_KOEL     | COMPANY | Kirloskar Oil Engines | KIRLOSENG
C_HBLENGINE | COMPANY | HBL Engineering | HBLENGINE | industrial batteries
C_APARINDS | COMPANY | Apar Industries | APARINDS | transformer oil + conductors + cables
C_HPL      | COMPANY | HPL Electric & Power | HPL
```
```
P_DATACENTRE -[NEEDS]-> P_TRANSFORMER
P_DATACENTRE -[NEEDS]-> P_SWITCHGEAR
P_DATACENTRE -[NEEDS]-> P_UPS
P_DATACENTRE -[NEEDS]-> P_GENSET
C_HITACHIENER -[SUPPLIES_TO {product:"HV transformers, grid connection"}]-> P_DATACENTRE
C_ABB -[SUPPLIES_TO {product:"switchgear, drives, high-efficiency motors for cooling loops"}]-> P_DATACENTRE
C_SCHNEIDER -[SUPPLIES_TO {product:"MV/LV distribution, UPS, prefabricated power skids"}]-> P_DATACENTRE
C_CGPOWER -[SUPPLIES_TO]-> P_DATACENTRE
C_VOLTAMP -[SUPPLIES_TO {product:"dry-type + oil-filled distribution transformers"}]-> P_DATACENTRE
C_CUMMINSIND -[SUPPLIES_TO {product:"HHP diesel gensets, N+1 backup"}]-> P_DATACENTRE
C_APARINDS -[SUPPLIES_TO {product:"transformer oil, specialty oils, elastomeric cables"}]-> P_TRANSFORMER
C_TRIL -[CONSTRAINED_BY]-> CN_CRGO
C_VOLTAMP -[CONSTRAINED_BY]-> CN_CRGO
```
> **Cost-arbitrage node worth flagging:** a 500 MVA transformer is quoted around $2.5m in India vs $14–22m in the US. That spread is why Indian transformer makers are being pulled into *global* AI supply chains, not just domestic ones — an export optionality edge that does not depend on Indian DC capacity at all.

#### 3.3.3 Cooling & thermal management

```
C_BLUESTARCO | COMPANY | Blue Star | BLUESTARCO | precision cooling + MEP/EPC for DCs
C_VOLTAS   | COMPANY | Voltas | VOLTAS | screw/centrifugal/oil-free chillers
C_KRN      | COMPANY | KRN Heat Exchanger | KRN | heat exchangers, ~6x capacity expansion
C_AMBER    | COMPANY | Amber Enterprises | AMBER | HVAC components, contract manufacturing
C_JCHAC    | COMPANY | Johnson Controls-Hitachi AC India | JCHAC
C_THERMAX  | COMPANY | Thermax | THERMAX | chillers, absorption cooling, water treatment
C_EPACK    | COMPANY | EPACK Durable | EPACK
```
```
P_DATACENTRE -[NEEDS {intensity:"rising with rack density"}]-> P_COOLING
T_LIQUIDCOOL -[CAUSES {demand_for:"CDUs, heat exchangers, coolant loops"}]-> P_COOLING
C_KRN -[SUPPLIES_TO {note:"supplies Schneider, Climaveneta; targeting ~50% share of India DC heat exchangers; Google Vizag campus sized ~Rs 1,500cr opportunity"}]-> C_SCHNEIDER
C_BLUESTARCO -[SUPPLIES_TO {product:"precision AC, chillers, MEP execution"}]-> P_DATACENTRE
C_VOLTAS -[SUPPLIES_TO {product:"chillers"}]-> P_DATACENTRE
C_THERMAX -[SUPPLIES_TO {product:"chillers + water treatment + captive power"}]-> P_DATACENTRE
```
> **Purity test:** among cooling names, KRN is the one where data centres are the *primary* revenue driver rather than one segment among several. Blue Star, Voltas, Amber and Thermax are diluted by room AC, projects and process businesses. This is the difference between a thematic proxy and a thematic pure-play, and it should govern position sizing.

#### 3.3.4 Compute hardware, EMS & semiconductors

```
C_NETWEB   | COMPANY | Netweb Technologies | NETWEB | Tyrone AI servers on NVIDIA Blackwell
C_DIXON    | COMPANY | Dixon Technologies | DIXON | EMS scale-up into IT hardware/servers
C_KAYNES   | COMPANY | Kaynes Technology | KAYNES | EMS + OSAT (~Rs 2,800cr Telangana)
C_SYRMA    | COMPANY | Syrma SGS Technology | SYRMA
C_CYIENTDLM | COMPANY | Cyient DLM | CYIENTDLM
C_TATAELXSI | COMPANY | Tata Elxsi | TATAELXSI
C_MOSCHIP  | COMPANY | MosChip Technologies | MOSCHIP | chip design services
C_SPEL     | COMPANY | SPEL Semiconductor | (BSE) | assembly/test
PC_NVIDIA  | PRIVATE_CO | NVIDIA | GPU supply
PC_TATAELEC | PRIVATE_CO | Tata Electronics | fab + OSAT
```
```
P_AISERVER -[NEEDS]-> RM_GPU
P_AISERVER -[NEEDS]-> CN_HBM
C_NETWEB -[PRODUCES]-> P_AISERVER
C_NETWEB -[SUPPLIES_TO {order:"Rs 1,734cr Blackwell servers, execution Q4FY26–H1FY27"}]-> CU_INDIAAI
C_NETWEB -[CONSTRAINED_BY {severity:critical}]-> CN_GPU
C_DIXON -[BENEFITS_FROM {order:1, channel:"server assembly under IT hardware PLI"}]-> POL_PLI_IT
C_KAYNES -[BENEFITS_FROM {order:1, channel:"PCBA + OSAT for power and networking gear"}]-> POL_SEMICON
C_CGPOWER -[BENEFITS_FROM {order:1, channel:"OSAT JV"}]-> POL_SEMICON
PC_NVIDIA -[SUPPLIES_TO {gating:true}]-> C_NETWEB
```
> **Nuance:** Indian AI server "manufacturing" is largely **integration and assembly around imported GPUs**. Gross margins are structurally thin relative to the multiple the market often applies, and revenue is lumpy and tender-driven. The `CN_GPU` edge is a hard gate — order books can be won and still not convert on schedule.

#### 3.3.5 Connectivity, fibre & networking

```
C_STLTECH  | COMPANY | Sterlite Technologies | STLTECH | optical fibre + cable
C_HFCL     | COMPANY | HFCL | HFCL
C_TEJASNET | COMPANY | Tejas Networks | TEJASNET
C_BIRLACABLE | COMPANY | Birla Cable | BIRLACABLE
C_VINDHYATEL | COMPANY | Vindhya Telelinks | VINDHYATEL
C_INDUSTOWER | COMPANY | Indus Towers | INDUSTOWER | edge compute at tower sites
```
```
P_DATACENTRE -[NEEDS]-> P_FIBRE
C_STLTECH -[SUPPLIES_TO {product:"optical fibre cable, DC interconnect"}]-> P_DATACENTRE
C_TATACOMM -[SUPPLIES_TO {product:"subsea capacity, DC interconnect"}]-> CU_HYPERSCALER
C_INDUSTOWER -[BENEFITS_FROM {order:3, channel:"edge inference nodes co-located at tower sites"}]-> T_GENAI
```

#### 3.3.6 Cables & power distribution

```
C_POLYCAB  | COMPANY | Polycab India | POLYCAB
C_KEI      | COMPANY | KEI Industries | KEI
C_RRKABEL  | COMPANY | R R Kabel | RRKABEL
C_FINCABLES | COMPANY | Finolex Cables | FINCABLES
C_SKIPPER  | COMPANY | Skipper | SKIPPER | transmission towers
```
```
P_DATACENTRE -[NEEDS]-> P_POWERCABLE
C_POLYCAB -[SUPPLIES_TO]-> P_DATACENTRE
C_KEI -[SUPPLIES_TO {product:"HT/EHV cables"}]-> P_DATACENTRE
P_POWERCABLE -[NEEDS]-> RM_COPPER
P_POWERCABLE -[NEEDS]-> RM_PVC
```

#### 3.3.7 EPC, construction & real estate

```
C_LT       | COMPANY | Larsen & Toubro | LT | DC EPC + electrical + civil
C_TECHNOE  | COMPANY | Techno Electric & Engineering | TECHNOE | T&D EPC + own DC foray
C_KALPATPOWR | COMPANY | Kalpataru Projects | KPIL
C_KEC      | COMPANY | KEC International | KEC
C_AHLUCONT | COMPANY | Ahluwalia Contracts | AHLUCONT
C_PSPPROJECT | COMPANY | PSP Projects | PSPPROJECT
C_DLF      | COMPANY | DLF | DLF
C_EMBASSY  | COMPANY | Embassy Office Parks REIT | EMBASSY
C_MINDSPACE | COMPANY | Mindspace Business Parks REIT | MINDSPACE
C_BRIGADE  | COMPANY | Brigade Enterprises | BRIGADE
```
```
C_LT -[SUPPLIES_TO {product:"turnkey DC shell + MEP + substation"}]-> P_DATACENTRE
C_TECHNOE -[BENEFITS_FROM {order:1, channel:"substation EPC for DC power delivery + own DC assets"}]-> E_AIBUILD
C_KEC -[SUPPLIES_TO {product:"transmission lines to DC clusters"}]-> CN_TRANSMISSION
C_EMBASSY -[BENEFITS_FROM {order:3, channel:"GCC/AI-driven office absorption in Bengaluru/Hyderabad"}]-> CU_GCC_ENT
```

#### 3.3.8 Power generation, transmission & grid

```
C_NTPC     | COMPANY | NTPC | NTPC
C_NTPCGREEN | COMPANY | NTPC Green Energy | NTPCGREEN
C_POWERGRID | COMPANY | Power Grid Corporation | POWERGRID
C_TATAPOWER | COMPANY | Tata Power | TATAPOWER
C_ADANIPOWER | COMPANY | Adani Power | ADANIPOWER
C_ADANIGREEN | COMPANY | Adani Green Energy | ADANIGREEN
C_JSWENERGY | COMPANY | JSW Energy | JSWENERGY
C_TORNTPOWER | COMPANY | Torrent Power | TORNTPOWER
C_NHPC     | COMPANY | NHPC | NHPC
C_IEX      | COMPANY | Indian Energy Exchange | IEX
C_SUZLON   | COMPANY | Suzlon Energy | SUZLON
C_INOXWIND | COMPANY | Inox Wind | INOXWIND
C_KPIGREEN | COMPANY | KPI Green Energy | KPIGREEN
C_BHEL     | COMPANY | BHEL | BHEL
```
```
P_DATACENTRE -[NEEDS {quantum:"26.3 GW incremental by FY32"}]-> P_FIRMPOWER
C_NTPCGREEN -[SUPPLIES_TO {deal:"~2 GW renewable MoU with CtrlS"}]-> PC_CTRLS
C_POWERGRID -[BENEFITS_FROM {order:1, channel:"ISTS buildout; ~Rs 9.16 lakh crore transmission plan"}]-> CN_TRANSMISSION
C_TATAPOWER -[BENEFITS_FROM {order:1, channel:"round-the-clock RE + distribution licence in Mumbai DC cluster"}]-> E_AIBUILD
C_NHPC -[BENEFITS_FROM {order:2, channel:"hydro provides flexible peaking for volatile DC load ramps"}]-> CN_GRIDSTAB
C_IEX -[BENEFITS_FROM {order:2, channel:"DC operators trading RTC/green attributes on exchange"}]-> E_AIBUILD
```
> **Nuance:** government intent is that this load be *"primarily served by renewable energy capacity."* But renewables are variable and DC load is spiky, which is precisely why `CN_GRIDSTAB` exists as a node — and why BESS, STATCOMs and synchronous condensers are the highest-quality second-order derivative in the whole theme.

---

### 3.4 SECOND ORDER — inputs to the first-order suppliers

#### 3.4.1 Metals & electrical materials

```
RM_COPPER  | RAW_MATERIAL | Copper | windings, busbars, cables
RM_ALUMINIUM | RAW_MATERIAL | Aluminium | conductors, busbars, heat sinks
RM_CRGO_STEEL | RAW_MATERIAL | CRGO electrical steel | imported, the true transformer bottleneck
RM_TRANSFORMEROIL | RAW_MATERIAL | Transformer oil | base-oil derivative
```
```
C_HINDALCO -[SUPPLIES_TO {product:"aluminium conductors, busbars, copper (Birla Copper)"}]-> P_TRANSFORMER
C_HINDCOPPER -[SUPPLIES_TO]-> RM_COPPER
C_NALCO -[SUPPLIES_TO]-> RM_ALUMINIUM
C_APARINDS -[SUPPLIES_TO {product:"transformer oil"}]-> P_TRANSFORMER
C_RAMRATNA -[SUPPLIES_TO {product:"winding wire"}]-> P_TRANSFORMER
C_PRECWIRE -[SUPPLIES_TO {product:"enamelled copper winding wire"}]-> P_TRANSFORMER
CN_CRGO -[CONSTRAINS {note:"no meaningful domestic CRGO capacity; import-gated"}]-> C_VOLTAMP
```
```
C_HINDCOPPER | COMPANY | Hindustan Copper | HINDCOPPER
C_RAMRATNA   | COMPANY | Ram Ratna Wires | RAMRAT
C_PRECWIRE   | COMPANY | Precision Wires India | PRECWIRE
```

#### 3.4.2 Refrigerants, chemicals & thermal materials

```
C_GFL      | COMPANY | Gujarat Fluorochemicals | FLUOROCHEM | refrigerant gases, PTFE
C_SRF      | COMPANY | SRF | SRF | refrigerants, chemicals, films
C_NAVINFLUOR | COMPANY | Navin Fluorine | NAVINFLUOR
```
```
P_CHILLER -[NEEDS]-> RM_REFRIGERANT
C_GFL -[SUPPLIES_TO {product:"HFC/HFO refrigerants, immersion-cooling fluid optionality"}]-> P_CHILLER
C_SRF -[SUPPLIES_TO {product:"refrigerant gases"}]-> P_CHILLER
T_LIQUIDCOOL -[CAUSES {demand_for:"dielectric/immersion fluids"}]-> RM_REFRIGERANT
```

#### 3.4.3 Energy storage & grid stability

```
C_EXIDE    | COMPANY | Exide Industries | EXIDEIND | Li-ion gigafactory
C_AMARAJA  | COMPANY | Amara Raja Energy & Mobility | ARE&M
C_HBLENGINE | COMPANY | HBL Engineering | HBLENGINE
C_SERVOTECH | COMPANY | Servotech Renewable Power | SERVOTECH
C_WAAREE   | COMPANY | Waaree Energies | WAAREEENER | modules + BESS
```
```
CN_GRIDSTAB -[CAUSES {demand_for:"BESS"}]-> C_EXIDE
CN_GRIDSTAB -[CAUSES {demand_for:"STATCOM, synchronous condensers"}]-> C_HITACHIENER
P_UPS -[NEEDS]-> RM_BATTERY
C_HBLENGINE -[SUPPLIES_TO {product:"industrial VRLA/Ni-Cd for DC backup"}]-> P_UPS
```

#### 3.4.4 Water, environment & building services

```
C_VATECHWABAG | COMPANY | VA Tech Wabag | WABAG | water & wastewater treatment
C_IONEXCHANGE | COMPANY | Ion Exchange India | IONEXCHANG
C_THERMAX  | COMPANY | Thermax | THERMAX
```
```
CN_WATER -[CAUSES {demand_for:"ZLD, recycling, cooling-water treatment"}]-> C_VATECHWABAG
C_IONEXCHANGE -[SUPPLIES_TO {product:"DM water, cooling tower chemistry"}]-> P_DATACENTRE
```
> This branch is under-covered relative to its logical necessity: over half of India's DC capacity sits in water-stressed regions, and consumption is projected to rise from ~150bn to ~358bn litres/year. Water treatment is a *mandatory* line item, not a discretionary one.

#### 3.4.5 Construction materials

```
C_ULTRACEMCO -[SUPPLIES_TO {product:"cement for DC shells, ~Rs 400–600cr per 100 MW campus"}]-> P_DATACENTRE
C_JSWSTEEL -[SUPPLIES_TO {product:"structural steel, rebar"}]-> P_DATACENTRE
```

---

### 3.5 THIRD ORDER — financing, second-derivative demand, and disruption risk

#### 3.5.1 Capital providers

```
C_PFC      | COMPANY | Power Finance Corporation | PFC
C_RECLTD   | COMPANY | REC Ltd | RECLTD
C_IREDA    | COMPANY | IREDA | IREDA
C_SBIN     | COMPANY | State Bank of India | SBIN
```
```
C_PFC -[BENEFITS_FROM {order:3, channel:"lending to generation + transmission serving DC load"}]-> CN_POWER
C_IREDA -[BENEFITS_FROM {order:3, channel:"RE + BESS project finance for DC PPAs"}]-> CN_GRIDSTAB
C_SBIN -[BENEFITS_FROM {order:3, channel:"large-ticket infra credit demand revival"}]-> E_AIBUILD
```

#### 3.5.2 Talent, services & the ambiguous IT-services node

```
C_TEAMLEASE | COMPANY | TeamLease Services | TEAMLEASE
C_QUESS    | COMPANY | Quess Corp | QUESS
C_TCS / C_INFY / C_HCLTECH / C_WIPRO / C_LTIM  (see §2.5.2)
C_PERSISTENT | COMPANY | Persistent Systems | PERSISTENT
C_COFORGE  | COMPANY | Coforge | COFORGE
C_QUICKHEAL | COMPANY | Quick Heal Technologies | QUICKHEAL
```
```
C_TEAMLEASE -[BENEFITS_FROM {order:3, channel:"DC construction + O&M staffing, HV/liquid-cooling skills gap"}]-> CN_SKILLS
C_TCS -[HEDGES {note:"AI-led productivity deflation compresses the linear headcount model (−) while GCC buildout and AI migration deals expand TAM (+); direction is genuinely contested"}]-> T_GENAI
C_QUICKHEAL -[BENEFITS_FROM {order:3, channel:"DPDP Act compliance + data localisation"}]-> POL_DC_TAX
```
> This is the one branch where the sign is honestly **unresolved**. Treat any confident claim that AI is unambiguously good or bad for Indian IT services with suspicion; the same technology expands the addressable market and attacks the pricing model at once.

#### 3.5.3 Second-derivative real economy

```
C_INDUSTOWER -[BENEFITS_FROM {order:3, channel:"edge inference"}]-> T_GENAI
C_IEX -[BENEFITS_FROM {order:3}]-> CN_POWER
C_ADANIPORTS -[BENEFITS_FROM {order:3, channel:"import volumes of transformers, chillers, servers, GPUs"}]-> E_AIBUILD
C_INDIGO -[BENEFITS_FROM {order:3, channel:"air-freighted GPU/server cargo + business travel to DC clusters"}]-> E_AIBUILD
C_ANANTRAJ -[BENEFITS_FROM {order:3, channel:"land value re-rating in DC clusters"}]-> CN_LAND
```

---

## 4. CROSS-THEME NODES — where the two graphs touch

These are the highest-information nodes in the file. A node with edges into both themes has a **non-obvious net exposure** that single-theme screens systematically miss.

### 4.1 Companies with edges into both themes

| Node | Theme A edge | Theme B edge | Net character |
|---|---|---|---|
| **C_RIL** (Reliance) | O2C feedstock cost, crude sourcing, LPG marketing (−); GRM expansion, KG-D6 gas realisation (+) | Jamnagar AI campus, Meta 168 MW lease, ~$110bn commitment (+) | The single most bi-thematic node in the Indian market. Energy shock funds the AI capex; AI capex is where the energy cash flow goes. |
| **C_ADANIENT / Adani complex** | Adani Ports cargo mix, Adani Total Gas LNG cost (−) | AdaniConneX 2→5 GW, ~$100bn by 2035 (+) | Same structure as Reliance: an energy/infra balance sheet converting into digital infrastructure. |
| **C_HINDALCO** | Energy-intensive smelting, CP coke, freight (−) | Copper + aluminium into cables, busbars, transformers (+) | Cost hit and demand boost arrive simultaneously. Net depends on LME pass-through. |
| **C_APARINDS** | Transformer oil is a **base-oil derivative** — directly crude-linked (−) | Transformer oil + conductors + elastomeric cables into DC/T&D buildout (+) | The cleanest single-company illustration of the overlap: one product, both themes, opposite signs. |
| **C_POLYCAB / C_KEI** | PVC and polymer compounds are crude derivatives (−); copper is the larger input | DC + T&D cable demand (+) | Demand edge generally dominates cost edge in an up-capex cycle. |
| **C_SRF / C_GFL / C_NAVINFLUOR** | Crude-linked feedstock and energy (−) | Refrigerants and fluoropolymers into chillers and liquid cooling (+) | |
| **C_THERMAX** | Refinery/process capex is oil-price-linked (+/−) | DC chillers, captive power, water treatment (+) | |
| **C_LT** | GCC order book exposure to conflict (−), to high oil (+) | DC EPC, substations, electricals (+) | Genuinely three-directional. |
| **C_VOLTAS** | Legacy Gulf MEP project exposure (−) | Chillers for DC cooling (+) | |
| **C_CUMMINSIND / C_KOEL** | Diesel price affects genset running economics (−) | Backup gensets are mandatory DC capex (+) | Capex demand >> opex sensitivity. |
| **C_ULTRACEMCO / C_JSWSTEEL** | Pet coke, coal, freight (−) | DC shell construction volume (+) | The volume boost is small relative to the cost hit. |
| **C_ADANIPORTS** | Gulf trade-lane disruption (−) | Import volumes of DC equipment (+) | |
| **C_INDIGO** | ATF, INR, airspace (−−−) | Air-freighted GPU/server cargo (+, trivial) | Overlap exists but is not material — an example of a **weak edge that should be pruned** when scoring. |
| **C_NTPC / C_COALINDIA** | Gas-to-coal substitution (+) | DC baseload demand (+) | Rare double-positive. |
| **C_IREDA / C_PFC** | Energy-security capex push (+) | DC power + BESS financing (+) | Double-positive via policy. |

### 4.2 Cross-theme mechanism edges (theme-to-theme, not company-to-company)

```
E_HORMUZ -[RAISES_COST_OF]-> P_FIRMPOWER
P_FIRMPOWER -[IS_50_PCT_OF]-> P_DATACENTRE_OPEX
∴ E_HORMUZ -[HURT_BY {order:2, path:"LNG price → captive/grid power tariff → DC opex"}]-> E_AIBUILD

E_HORMUZ -[CAUSES {+}]-> M_WARRISK
M_WARRISK -[RAISES]-> COST_OF_IMPORTED_CAPEX
∴ E_HORMUZ -[DELAYS {path:"freight + war-risk premia on imported transformers, chillers, GPUs"}]-> E_AIBUILD

E_HORMUZ -[CAUSES {+}]-> M_INR
M_INR -[RAISES]-> CN_GPU_LANDED_COST
∴ E_HORMUZ -[HURT_BY {order:3, path:"INR depreciation → dollar-denominated GPU/equipment capex inflates"}]-> C_NETWEB

E_HORMUZ -[CAUSES {+}]-> M_REPO
M_REPO -[RAISES]-> DC_PROJECT_WACC
∴ E_HORMUZ -[HURT_BY {order:3, path:"higher discount rate → long-duration DC and RE project IRRs compress"}]-> C_ADANIGREEN

E_AIBUILD -[CAUSES {+}]-> CN_POWER
CN_POWER -[CAUSES {+}]-> RM_COAL_DEMAND
∴ E_AIBUILD -[BENEFITS_FROM {order:3, path:"DC load met partly by coal despite RE intent"}]-> C_COALINDIA
```

> **The most important cross-theme statement in this file:** *an energy shock is a tax on the AI buildout.* Power is ~50% of data-centre operating cost, and a meaningful share of India's marginal generation and its imported capital equipment is exposed to Hormuz. A long Theme-B position is implicitly short Theme-A escalation. If you build a portfolio from this graph, that correlation is not diversification — it is a hidden factor bet.

---

## 5. Traversal recipes — how to actually use this graph

**Recipe 1 — Find true 2nd-order exposure to a crude spike, ranked by severity**
```
MATCH path = (e:EVENT {id:'E_HORMUZ'})-[:CAUSES]->(m:MACRO_VAR)
             -[:PRICE_LINKED_TO]-(rm:RAW_MATERIAL)<-[:NEEDS]-(p:PRODUCT)<-[:PRODUCES]-(c:COMPANY)
RETURN c, rm, length(path)
ORDER BY rm.cost_share_of_COGS DESC
```
Then filter by the two modifiers that determine whether the edge actually reaches earnings: **pricing power** and **inventory lag**.

**Recipe 2 — Find nodes where both themes hit the same P&L**
```
MATCH (c:COMPANY)
WHERE (c)-[:HURT_BY|BENEFITS_FROM|HEDGES]->(:EVENT {id:'E_HORMUZ'})
  AND (c)-[:HURT_BY|BENEFITS_FROM|SUPPLIES_TO|HEDGES]->(:EVENT {id:'E_AIBUILD'})
RETURN c
```
→ returns §4.1.

**Recipe 3 — Find the bottleneck plays (usually the best risk-reward in a capex theme)**
```
MATCH (cn:CONSTRAINT)<-[:CONSTRAINED_BY]-(:EVENT {id:'E_AIBUILD'})
MATCH (c:COMPANY)-[:SUPPLIES_TO|RELIEVES]->(cn)
RETURN cn, collect(c)
```
→ transformers/CRGO, grid stability (BESS/STATCOM), liquid cooling, water treatment, transmission EPC.

**Recipe 4 — Sign-flip test (de-escalation scenario)**
Invert every `sign` attribute on edges out of `E_HORMUZ` and re-run Recipe 1. Nodes that appear in the top decile of *both* the escalation and de-escalation runs are **volatility-exposed, not direction-exposed** — usually OMCs, shipping and CGD.

**Recipe 5 — Purity scoring**
For each company, compute:
```
theme_purity = (revenue attributable to theme) / (total revenue)
edge_count   = number of distinct paths from shock node
```
High purity + low edge count = a clean but fragile single-channel bet (e.g. KRN Heat Exchanger).
Low purity + high edge count = a diluted but resilient conglomerate proxy (e.g. Reliance, L&T).
Neither is better; they answer different questions.

---

## 6. Node validation checklist

Before treating any edge in this file as investable, confirm:

1. **Disclosed segment revenue** — does the company actually break out the exposure, or is it an analyst inference? Many "data centre stocks" have never quantified DC revenue.
2. **Order book conversion** — announced order ≠ recognised revenue. Netweb's Blackwell order, for instance, is gated by GPU lead times of 36–52 weeks.
3. **Cost share** — an input at 3% of COGS does not move earnings regardless of how cleanly the edge is drawn.
4. **Pass-through contract structure** — formula pricing, quarterly resets, or subsidy cost-plus can neutralise a large-looking commodity edge.
5. **Whether it is already priced** — an equal-weighted index of 28 Indian DC-supply-chain stocks rose ~50% in the first half of 2026. The graph tells you the *causal* link, not the *residual* opportunity.
6. **Regulatory clawback** — windfall tax on upstream, administered pricing on OMCs, subsidy timing on fertilisers.

---

## 7. Known gaps and extension hooks

Areas where this graph is deliberately thin and worth extending:

- **Unlisted-to-listed bridges.** CtrlS, Yotta, Nxtra and AirTrunk drive listed vendors' order books. Adding their capex schedules as `PRIVATE_CO` nodes with `SUPPLIES_TO` edges would materially improve Theme B's predictive value. Watch also for DC operator IPOs, which would convert proxy exposure into direct exposure.
- **CRGO and GPU import origins.** Both are hard gates with no domestic substitute; mapping supplier countries would create a third geopolitical theme node.
- **State-level nodes.** Andhra Pradesh, Maharashtra, Tamil Nadu and Telangana DC policies (power tariff, stamp duty, water allocation) differ enough to change project IRRs. Add `POL_STATE_*` nodes.
- **Time-decay on edges.** Hormuz edges are regime-dependent and should carry a `half_life` attribute; AI edges are structural and should carry a `duration` attribute. They are not the same kind of edge and should not be scored identically.
- **Water and land as tradeable constraints.** Currently modelled as `CONSTRAINT` nodes with few company edges; the listed water-treatment set is small but the necessity is absolute.
- **Second-round demand destruction.** The graph models cost shocks well and demand-destruction feedback loops poorly (e.g. high LPG → LPG-to-firewood switching → rural distribution volumes).

---

## 8. Source basis

Grounded on August 2026 reporting and primary data: IEA chokepoint analysis, Ministry of Power statements to Parliament (26.3 GW by FY32), MeitY/Cushman & Wakefield/Savills capacity data, ORF and CareEdge inflation and freight studies, UNCTAD fertiliser supply-chain warnings, Budget 2026-27 data-centre tax provisions, and company disclosures. Commodity prices, freight rates and Hormuz transit status were moving rapidly at the time of compilation — **re-validate all macro nodes before use.**
