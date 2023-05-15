OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-2.987342098445111) q[0];
rz(-0.5584455010549814) q[0];
ry(0.016873824589412223) q[1];
rz(1.62614116179717) q[1];
ry(-3.051074014800742) q[2];
rz(-1.160999999819161) q[2];
ry(-1.5917605260394903) q[3];
rz(1.5826947320888725) q[3];
ry(-1.5482771005863691) q[4];
rz(2.4769806431763213) q[4];
ry(0.90386286517966) q[5];
rz(0.4856572977684709) q[5];
ry(-1.5318460220960162) q[6];
rz(-3.01393645026532) q[6];
ry(-1.5862476418905096) q[7];
rz(-3.0993747817830903) q[7];
ry(3.1332367651938995) q[8];
rz(0.5508240893433581) q[8];
ry(-0.0051631739549575315) q[9];
rz(-0.059776578783908185) q[9];
ry(-0.8798724031877221) q[10];
rz(2.963883272510107) q[10];
ry(-1.5720426050980425) q[11];
rz(-0.0007518948415823347) q[11];
ry(-0.014008127546371261) q[12];
rz(1.9898706681106386) q[12];
ry(0.0007294693905173233) q[13];
rz(-2.0363868512519945) q[13];
ry(-1.5652002346411793) q[14];
rz(-0.7270144159512212) q[14];
ry(-1.5837240607483007) q[15];
rz(-0.7125382412829109) q[15];
ry(1.6461120764947303) q[16];
rz(1.973377310152757) q[16];
ry(-1.6767426109157988) q[17];
rz(-0.12841772807580654) q[17];
ry(-2.803597912759742) q[18];
rz(1.317939006781855) q[18];
ry(-1.089679300911004) q[19];
rz(1.3663427104676946) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.11248666829906107) q[0];
rz(-0.5372576636025929) q[0];
ry(3.0304853111779604) q[1];
rz(0.9927662342154887) q[1];
ry(-0.01068322323369575) q[2];
rz(1.2422044067843316) q[2];
ry(0.9990800630937331) q[3];
rz(2.5848336466150474) q[3];
ry(-0.012334584149309435) q[4];
rz(0.7221641090281778) q[4];
ry(-3.1410889308934085) q[5];
rz(0.6403729121255671) q[5];
ry(-1.177785480523397) q[6];
rz(-2.890588304830689) q[6];
ry(1.1735012871612989) q[7];
rz(2.5225307881918924) q[7];
ry(-0.4673607241548953) q[8];
rz(3.0889517523944723) q[8];
ry(-2.880259630311083) q[9];
rz(2.984056918251376) q[9];
ry(0.0024708595690922015) q[10];
rz(-2.423743382573656) q[10];
ry(-1.5510594599725043) q[11];
rz(-1.381275435472432) q[11];
ry(1.5707326495093437) q[12];
rz(-1.2566537584308466) q[12];
ry(-0.7154702796945624) q[13];
rz(-1.5421093600399112) q[13];
ry(-3.1179050506783947) q[14];
rz(0.8761212226306546) q[14];
ry(-3.117244958932053) q[15];
rz(-1.4716509407569012) q[15];
ry(-0.0009318564787422295) q[16];
rz(-0.7021844606321226) q[16];
ry(-3.1365499029347075) q[17];
rz(1.2513282509955082) q[17];
ry(2.996249367196631) q[18];
rz(-2.805185855640351) q[18];
ry(1.3594204557061023) q[19];
rz(2.29356757810155) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.3872307586475137) q[0];
rz(2.86685159646196) q[0];
ry(-3.115202128702117) q[1];
rz(-0.29852975037585855) q[1];
ry(-0.875647461712564) q[2];
rz(-1.9080002426786054) q[2];
ry(-1.4477147157784311) q[3];
rz(0.40224308069402426) q[3];
ry(-1.5461869997146571) q[4];
rz(2.2699351111540977) q[4];
ry(0.944824673605031) q[5];
rz(1.4736340551397458) q[5];
ry(-0.49327968493256935) q[6];
rz(2.468989294406447) q[6];
ry(-0.9713183281586373) q[7];
rz(-2.9385558834574237) q[7];
ry(-2.4952061549998383) q[8];
rz(-1.2494913515553945) q[8];
ry(-0.580579872653594) q[9];
rz(1.4923467585210266) q[9];
ry(1.5613717376747513) q[10];
rz(1.5766532361803423) q[10];
ry(1.5563619994585656) q[11];
rz(1.5796705563039222) q[11];
ry(-3.135324308789985) q[12];
rz(0.9426118497678919) q[12];
ry(0.4067367839127618) q[13];
rz(2.7434208554682478) q[13];
ry(-1.9649499840643534) q[14];
rz(1.950263937581749) q[14];
ry(0.838996919806009) q[15];
rz(-1.6640312873024805) q[15];
ry(-0.27311069669326665) q[16];
rz(-2.7097694412361193) q[16];
ry(-1.0115155529615385) q[17];
rz(0.2354589821230748) q[17];
ry(-1.3368095733526661) q[18];
rz(-1.3075982498981409) q[18];
ry(-0.9959261879764002) q[19];
rz(1.419900070815757) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.7684839975029732) q[0];
rz(0.1667394831157836) q[0];
ry(0.3791280807684528) q[1];
rz(1.3668306452651802) q[1];
ry(0.018176354503657108) q[2];
rz(3.0482218203715754) q[2];
ry(1.6545022920498986) q[3];
rz(-0.5346022058441271) q[3];
ry(3.133666338528965) q[4];
rz(1.4497076761682726) q[4];
ry(3.136912629449597) q[5];
rz(2.704884435363285) q[5];
ry(-3.100100248330565) q[6];
rz(2.813042808123508) q[6];
ry(-0.012755461027524008) q[7];
rz(-1.8285644434578063) q[7];
ry(-2.501839162789609) q[8];
rz(-1.0128074393436657) q[8];
ry(-0.758108240802499) q[9];
rz(0.7813446522036359) q[9];
ry(-1.5433136097869378) q[10];
rz(0.8772587653791375) q[10];
ry(-1.5643832289595743) q[11];
rz(1.1193308119713568) q[11];
ry(1.5788116036485311) q[12];
rz(-1.5710279880919549) q[12];
ry(-3.125205706401731) q[13];
rz(1.1485244608491054) q[13];
ry(-3.039705848564895) q[14];
rz(-2.6219311905797538) q[14];
ry(-0.04199661547787514) q[15];
rz(-0.8331043621115982) q[15];
ry(-1.5649948567402827) q[16];
rz(-0.054747776755227136) q[16];
ry(1.5511705340109194) q[17];
rz(1.9496041638006303) q[17];
ry(1.2219026294567312) q[18];
rz(1.8171210179319268) q[18];
ry(-1.3478186910207581) q[19];
rz(-1.2795662211792111) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.7733714780345249) q[0];
rz(1.2378788072799214) q[0];
ry(-2.793332830686384) q[1];
rz(-2.3191271192914527) q[1];
ry(-2.905994248201241) q[2];
rz(-1.6380452016217513) q[2];
ry(-1.2402073234932072) q[3];
rz(0.5420812149507714) q[3];
ry(0.06688732289525401) q[4];
rz(1.1267712204902) q[4];
ry(3.0300707857996514) q[5];
rz(0.7940026162112588) q[5];
ry(-3.004779004318608) q[6];
rz(-0.6122951426894964) q[6];
ry(2.920885453157864) q[7];
rz(-1.9824290594389333) q[7];
ry(0.1814035133797683) q[8];
rz(-2.0429987069325017) q[8];
ry(1.550605623034187) q[9];
rz(-1.7153431987267105) q[9];
ry(3.135896569875583) q[10];
rz(-2.454732470442539) q[10];
ry(-0.004367643237411793) q[11];
rz(1.851123920222206) q[11];
ry(1.8861284739770208) q[12];
rz(1.5668385676008196) q[12];
ry(-0.19375971570444506) q[13];
rz(0.15292738798245156) q[13];
ry(-0.0010429132150305504) q[14];
rz(3.129721735781509) q[14];
ry(3.1380715432633304) q[15];
rz(2.5275906966135198) q[15];
ry(1.4868742955606642) q[16];
rz(2.991858021146484) q[16];
ry(-1.6287356840338905) q[17];
rz(0.12199027491863693) q[17];
ry(-1.471272502266049) q[18];
rz(-0.4819915725432579) q[18];
ry(1.9514130876332914) q[19];
rz(0.6692216831643414) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.3966257426420734) q[0];
rz(0.14231261001756934) q[0];
ry(-2.8795770832879493) q[1];
rz(-0.585400278283549) q[1];
ry(-0.19768766049248931) q[2];
rz(3.116814280263077) q[2];
ry(-2.099239414900924) q[3];
rz(-1.1556179586335273) q[3];
ry(0.0034128544322062234) q[4];
rz(1.0139099066537014) q[4];
ry(3.1309829339806474) q[5];
rz(2.569201644252425) q[5];
ry(-0.0003101549299344697) q[6];
rz(1.8091004652003448) q[6];
ry(-3.1228986245230446) q[7];
rz(1.8799632784116764) q[7];
ry(-2.604498627617763) q[8];
rz(2.09354524760087) q[8];
ry(1.704170287166689) q[9];
rz(-1.291848471347111) q[9];
ry(-0.30660367346516176) q[10];
rz(0.03358079376544421) q[10];
ry(3.0978055028745364) q[11];
rz(3.118781758171278) q[11];
ry(-1.5717836713416211) q[12];
rz(2.660005947308532) q[12];
ry(-3.13699166861966) q[13];
rz(0.08108577810105062) q[13];
ry(-0.3448254684468415) q[14];
rz(-1.2260324663384585) q[14];
ry(1.2529954968119599) q[15];
rz(0.6930066915002033) q[15];
ry(-2.5069765088278655) q[16];
rz(-0.500614702562328) q[16];
ry(0.994784486861893) q[17];
rz(2.7045470046094224) q[17];
ry(1.7778972472895724) q[18];
rz(1.3476826607690624) q[18];
ry(0.881599844155788) q[19];
rz(-0.8563722995357184) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.9477104901288653) q[0];
rz(0.2855893882928455) q[0];
ry(-2.9783301392783357) q[1];
rz(-2.624939769060291) q[1];
ry(2.8874394709475597) q[2];
rz(-0.5149724564933322) q[2];
ry(1.5846996748301647) q[3];
rz(-3.0467286709237977) q[3];
ry(3.0610923768605254) q[4];
rz(-0.043630195708621346) q[4];
ry(3.0372382856485944) q[5];
rz(2.938317796714159) q[5];
ry(-0.582097313165732) q[6];
rz(2.232169643357654) q[6];
ry(-2.5716568740689345) q[7];
rz(-2.920366185743946) q[7];
ry(0.7601507447994731) q[8];
rz(-2.2314054863299324) q[8];
ry(0.9525147580691842) q[9];
rz(1.317398921640729) q[9];
ry(3.0530317820223263) q[10];
rz(-1.5350969227289126) q[10];
ry(-0.32478245451128895) q[11];
rz(-1.514619125676582) q[11];
ry(-1.5659395208642783) q[12];
rz(-3.0663168805309637) q[12];
ry(1.6204901295089986) q[13];
rz(0.007742337497782877) q[13];
ry(-0.05208098080587665) q[14];
rz(-2.3595616787951625) q[14];
ry(3.0355053447289997) q[15];
rz(2.802673434751142) q[15];
ry(0.007690624236440513) q[16];
rz(-2.533625776839401) q[16];
ry(-0.0102429382505278) q[17];
rz(-3.1072225752930542) q[17];
ry(2.5796334814279334) q[18];
rz(-1.8579230361103862) q[18];
ry(0.20809318151113043) q[19];
rz(0.637625915902717) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.22037824391739136) q[0];
rz(-2.178501404632969) q[0];
ry(-2.7966592116192968) q[1];
rz(-1.0516263857250028) q[1];
ry(1.5597647397753098) q[2];
rz(3.040018917136879) q[2];
ry(-2.31941078087531) q[3];
rz(-1.825377051798517) q[3];
ry(0.21670026745141335) q[4];
rz(-0.9549457157162493) q[4];
ry(3.098044038907911) q[5];
rz(-2.7635630353999483) q[5];
ry(1.5706848511588456) q[6];
rz(-3.110416432001595) q[6];
ry(-1.5444040445047973) q[7];
rz(-0.04662925050084957) q[7];
ry(-0.45159825419268174) q[8];
rz(0.6679151739190424) q[8];
ry(2.708358312291788) q[9];
rz(2.5462806447860356) q[9];
ry(-0.029033822605097015) q[10];
rz(1.579284080240396) q[10];
ry(0.16535643997353225) q[11];
rz(-1.585115261980528) q[11];
ry(1.5118176084015955) q[12];
rz(3.134318409244331) q[12];
ry(-1.5777228560781635) q[13];
rz(3.137345415796311) q[13];
ry(-0.10207810062147292) q[14];
rz(1.3471447649279265) q[14];
ry(-0.3673884013552886) q[15];
rz(-1.9294831686561151) q[15];
ry(-1.0832945242752476) q[16];
rz(2.562858960672271) q[16];
ry(-2.264315118488888) q[17];
rz(0.1683344510736076) q[17];
ry(1.91632157767412) q[18];
rz(0.08778521987412961) q[18];
ry(-2.2865765647640064) q[19];
rz(3.0289173364689033) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.3320158631600894) q[0];
rz(2.0887056370400474) q[0];
ry(-2.140207558555163) q[1];
rz(-0.11250532879970793) q[1];
ry(-0.0009660656454966996) q[2];
rz(-1.6961773419215325) q[2];
ry(-0.026734559774271638) q[3];
rz(-2.050604895347407) q[3];
ry(3.1353798816526064) q[4];
rz(2.123965823245684) q[4];
ry(-3.135377042513005) q[5];
rz(0.28770360731962336) q[5];
ry(1.566438031261487) q[6];
rz(-0.07137042611508401) q[6];
ry(1.5204630725923158) q[7];
rz(-0.39668463607132504) q[7];
ry(-3.0206942426406416) q[8];
rz(1.0198246676954605) q[8];
ry(-3.120462934400221) q[9];
rz(1.1205010312846944) q[9];
ry(1.5719929335525347) q[10];
rz(0.204201189775274) q[10];
ry(-1.5696606503479877) q[11];
rz(-2.099378458893044) q[11];
ry(1.5654879443263285) q[12];
rz(1.5901141839007966) q[12];
ry(1.580570327137183) q[13];
rz(1.7452338458223693) q[13];
ry(1.5739225630778755) q[14];
rz(-0.007494552603142246) q[14];
ry(-1.5691591524701078) q[15];
rz(3.1339482917864987) q[15];
ry(-3.1030145991491187) q[16];
rz(2.784047877371369) q[16];
ry(-3.13257286858709) q[17];
rz(-3.053447521041136) q[17];
ry(-0.3930816365643457) q[18];
rz(2.1748680237687803) q[18];
ry(0.5476744567436894) q[19];
rz(-0.7535373910626837) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.2031131410558764) q[0];
rz(-1.937318245820209) q[0];
ry(-1.5286623422346197) q[1];
rz(-0.29901990593258976) q[1];
ry(-2.4867561863972067) q[2];
rz(-0.7232597424731981) q[2];
ry(0.4434326432691998) q[3];
rz(0.5305166125244096) q[3];
ry(0.33904019053089307) q[4];
rz(-1.6244927672071656) q[4];
ry(2.1722061666230505) q[5];
rz(0.6786435811212059) q[5];
ry(-1.6032575543103957) q[6];
rz(3.0123412588184144) q[6];
ry(-1.7258267864780859) q[7];
rz(-1.6259921140380493) q[7];
ry(-0.3719578451981908) q[8];
rz(-2.8654207041288196) q[8];
ry(-0.4165175190287731) q[9];
rz(-2.1196773776581352) q[9];
ry(1.4093396277144192) q[10];
rz(0.9292238158973958) q[10];
ry(1.2023452850396883) q[11];
rz(-2.365938850192028) q[11];
ry(0.3026417377509624) q[12];
rz(-3.0317231451572924) q[12];
ry(1.5816103851089975) q[13];
rz(-1.0026439591236804) q[13];
ry(-1.560844859422423) q[14];
rz(3.017003119902114) q[14];
ry(1.575367298776869) q[15];
rz(1.967087197309895) q[15];
ry(-0.06343401540581557) q[16];
rz(0.27806054721433865) q[16];
ry(-0.06809084046522784) q[17];
rz(1.2237423403465548) q[17];
ry(-2.969489401997905) q[18];
rz(2.861913938645773) q[18];
ry(2.878299585841104) q[19];
rz(2.939344751369926) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.210678264432) q[0];
rz(-0.08911407943948291) q[0];
ry(1.6332516625859326) q[1];
rz(1.1733541031655363) q[1];
ry(0.03274514687959208) q[2];
rz(-2.3337808930961947) q[2];
ry(-3.1036391700872925) q[3];
rz(-2.1890771319896456) q[3];
ry(1.5732131348702136) q[4];
rz(-0.0015560469867213801) q[4];
ry(-1.5831342323736386) q[5];
rz(0.010501368993252491) q[5];
ry(2.922231707008491) q[6];
rz(-1.497831213761669) q[6];
ry(3.075408320621191) q[7];
rz(0.2336510437763677) q[7];
ry(-2.982211290655624) q[8];
rz(0.5671416404333481) q[8];
ry(-2.626811181118698) q[9];
rz(2.0949667572589425) q[9];
ry(1.5637201149559292) q[10];
rz(-3.133720422787031) q[10];
ry(-1.548033903050517) q[11];
rz(3.123601734991809) q[11];
ry(-2.7832468851991816) q[12];
rz(-2.534047137189027) q[12];
ry(0.05922108229075818) q[13];
rz(-1.5953834812006313) q[13];
ry(2.0024788850469766) q[14];
rz(0.6608394057877486) q[14];
ry(-0.011663852283347) q[15];
rz(1.1721819211683302) q[15];
ry(-1.5175435137866282) q[16];
rz(-1.9952132297442713) q[16];
ry(-2.2680989919348766) q[17];
rz(-0.023522506787944802) q[17];
ry(2.46944941524089) q[18];
rz(-0.2087359990622477) q[18];
ry(-0.6759376514305897) q[19];
rz(-0.14377883823908721) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.696045704170059) q[0];
rz(2.0957747284878696) q[0];
ry(-0.3238791356325441) q[1];
rz(2.254712781810425) q[1];
ry(0.0019129483640787368) q[2];
rz(-0.16067366163754127) q[2];
ry(-3.1358610765369797) q[3];
rz(1.652448137749051) q[3];
ry(-1.574232914995524) q[4];
rz(0.8512174296564057) q[4];
ry(1.5679164369889513) q[5];
rz(-0.08534222551002765) q[5];
ry(3.1092951305479626) q[6];
rz(1.3757289128746097) q[6];
ry(-0.023077653483064228) q[7];
rz(1.1928678293906205) q[7];
ry(-3.1307636708437077) q[8];
rz(-2.799662553826112) q[8];
ry(-2.9236386034487802) q[9];
rz(-0.9292935310481659) q[9];
ry(-1.5739832036193488) q[10];
rz(0.06339310271985886) q[10];
ry(1.5109380713965168) q[11];
rz(0.05945336890525207) q[11];
ry(-0.007458948223424375) q[12];
rz(-0.47795313419860375) q[12];
ry(-3.1375521858072513) q[13];
rz(2.177831930209794) q[13];
ry(-0.004666283861741504) q[14];
rz(-0.5992514416571866) q[14];
ry(3.109401723524932) q[15];
rz(-0.010622187806490047) q[15];
ry(0.018986112258162713) q[16];
rz(-1.1241800433512905) q[16];
ry(-3.1127749108055687) q[17];
rz(0.6165114673690129) q[17];
ry(-1.560955644840767) q[18];
rz(-1.7750008843592742) q[18];
ry(3.0907476761758783) q[19];
rz(2.712293598870956) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.0283373287730595) q[0];
rz(-2.6381679912979488) q[0];
ry(-2.60837165451645) q[1];
rz(-1.9095557110628567) q[1];
ry(1.5782929147065374) q[2];
rz(3.013087393951837) q[2];
ry(-1.7334411296064607) q[3];
rz(2.6138455019738434) q[3];
ry(-2.1088669114851726) q[4];
rz(-1.0267889593989548) q[4];
ry(1.7148423766170406) q[5];
rz(-0.9470754056089782) q[5];
ry(2.688305313275351) q[6];
rz(-2.3932483294308744) q[6];
ry(2.989589328667053) q[7];
rz(-1.1242292989238276) q[7];
ry(-0.608606212643859) q[8];
rz(-2.6216395600625413) q[8];
ry(2.406033107878831) q[9];
rz(1.1777134699675722) q[9];
ry(2.1681720336413424) q[10];
rz(0.9919555129191746) q[10];
ry(1.6550776691609785) q[11];
rz(0.17661377747686657) q[11];
ry(-0.7436596172059318) q[12];
rz(-2.28297882761608) q[12];
ry(0.3304489888758844) q[13];
rz(0.7910692487699158) q[13];
ry(-0.8074830706315463) q[14];
rz(-2.4428876787255196) q[14];
ry(-1.6104965507803886) q[15];
rz(-2.247006988079109) q[15];
ry(1.6542427873481989) q[16];
rz(2.44987235625124) q[16];
ry(-1.511436146322156) q[17];
rz(2.416662138058412) q[17];
ry(-1.2018195148503201) q[18];
rz(-1.8849843086322877) q[18];
ry(3.11065154577946) q[19];
rz(0.09841546432060856) q[19];