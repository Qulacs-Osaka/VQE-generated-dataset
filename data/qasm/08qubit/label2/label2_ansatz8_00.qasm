OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.90793460629288) q[0];
ry(-1.7899408449124792) q[1];
cx q[0],q[1];
ry(0.42528722403282376) q[0];
ry(0.7624720487096468) q[1];
cx q[0],q[1];
ry(-2.9165652971839195) q[2];
ry(-0.4257045867412399) q[3];
cx q[2],q[3];
ry(2.5094919087515746) q[2];
ry(-0.738897679930793) q[3];
cx q[2],q[3];
ry(-0.7769956873966288) q[4];
ry(-1.9931687811860546) q[5];
cx q[4],q[5];
ry(-0.30936256907379217) q[4];
ry(-0.08208477287516036) q[5];
cx q[4],q[5];
ry(-2.0447478533347248) q[6];
ry(1.384935988180648) q[7];
cx q[6],q[7];
ry(-0.7393677188546439) q[6];
ry(0.25534259309130913) q[7];
cx q[6],q[7];
ry(-1.0343946897182956) q[0];
ry(-1.7193674619446342) q[2];
cx q[0],q[2];
ry(-3.061833041325575) q[0];
ry(3.0475638955705073) q[2];
cx q[0],q[2];
ry(2.565498417603454) q[2];
ry(-1.1276251354739542) q[4];
cx q[2],q[4];
ry(2.8799386307555284) q[2];
ry(-3.098918789715415) q[4];
cx q[2],q[4];
ry(2.2377532864501672) q[4];
ry(1.1180326790463586) q[6];
cx q[4],q[6];
ry(2.578164439881615) q[4];
ry(3.0775371224462487) q[6];
cx q[4],q[6];
ry(-0.7276009625599995) q[1];
ry(0.8164462558689242) q[3];
cx q[1],q[3];
ry(-0.2279015076340249) q[1];
ry(1.005749180776454) q[3];
cx q[1],q[3];
ry(1.0991692336999304) q[3];
ry(-2.321874098683404) q[5];
cx q[3],q[5];
ry(2.86883593832519) q[3];
ry(3.1086215392673884) q[5];
cx q[3],q[5];
ry(0.2885835404552903) q[5];
ry(-2.8414248866661915) q[7];
cx q[5],q[7];
ry(-2.566643468028819) q[5];
ry(-3.138653993628412) q[7];
cx q[5],q[7];
ry(2.6205632965431516) q[0];
ry(1.5433399437540798) q[1];
cx q[0],q[1];
ry(-1.0282792127460239) q[0];
ry(-2.9330204414790293) q[1];
cx q[0],q[1];
ry(1.3331943528259391) q[2];
ry(-1.4522151691007776) q[3];
cx q[2],q[3];
ry(-2.927614589486021) q[2];
ry(-0.5002925031935357) q[3];
cx q[2],q[3];
ry(0.04440551085098842) q[4];
ry(-0.30602192256066674) q[5];
cx q[4],q[5];
ry(2.9733516480807154) q[4];
ry(-1.3938163015322882) q[5];
cx q[4],q[5];
ry(-1.3324251242531737) q[6];
ry(3.0889408971044388) q[7];
cx q[6],q[7];
ry(-1.2101452339197363) q[6];
ry(3.010413695420045) q[7];
cx q[6],q[7];
ry(1.6030348159218528) q[0];
ry(-0.4582920406627711) q[2];
cx q[0],q[2];
ry(0.03583743591445021) q[0];
ry(0.12345253859607234) q[2];
cx q[0],q[2];
ry(-1.3809251642673712) q[2];
ry(0.38204212448863123) q[4];
cx q[2],q[4];
ry(3.105333821249036) q[2];
ry(3.0527200627686164) q[4];
cx q[2],q[4];
ry(2.4507810672306345) q[4];
ry(-0.9706041372785945) q[6];
cx q[4],q[6];
ry(2.7572710124448006) q[4];
ry(3.1375982768597233) q[6];
cx q[4],q[6];
ry(-0.49455969203752126) q[1];
ry(1.956064282249712) q[3];
cx q[1],q[3];
ry(3.0845284637340757) q[1];
ry(0.16833609399663807) q[3];
cx q[1],q[3];
ry(0.05703317773981113) q[3];
ry(-1.2964224376084292) q[5];
cx q[3],q[5];
ry(0.0019964695733802884) q[3];
ry(-3.0732086038055764) q[5];
cx q[3],q[5];
ry(0.42343919452083084) q[5];
ry(-1.726641435939478) q[7];
cx q[5],q[7];
ry(-3.0879385714205254) q[5];
ry(-3.084508988711183) q[7];
cx q[5],q[7];
ry(-2.4966474257718687) q[0];
ry(1.521475270466059) q[1];
cx q[0],q[1];
ry(2.224329178371865) q[0];
ry(-2.620545313277562) q[1];
cx q[0],q[1];
ry(0.2060258226967795) q[2];
ry(-1.7951506689122878) q[3];
cx q[2],q[3];
ry(2.4285435797429615) q[2];
ry(1.3507442743703937) q[3];
cx q[2],q[3];
ry(-0.13953497112404153) q[4];
ry(-1.8755979255809463) q[5];
cx q[4],q[5];
ry(-0.5042272201158056) q[4];
ry(-0.11196113964319654) q[5];
cx q[4],q[5];
ry(-2.7262275716625557) q[6];
ry(-1.66330711691183) q[7];
cx q[6],q[7];
ry(1.2833717136599045) q[6];
ry(1.5035652235461) q[7];
cx q[6],q[7];
ry(-2.8898892363483135) q[0];
ry(-2.1190751165829695) q[2];
cx q[0],q[2];
ry(-3.1026433987481945) q[0];
ry(3.12455404797736) q[2];
cx q[0],q[2];
ry(1.4095202116094416) q[2];
ry(-1.484850420110439) q[4];
cx q[2],q[4];
ry(3.1346995431281672) q[2];
ry(3.1081628757227575) q[4];
cx q[2],q[4];
ry(-1.8244119884911338) q[4];
ry(1.7128489090024024) q[6];
cx q[4],q[6];
ry(-2.8557876458285993) q[4];
ry(3.122771482461081) q[6];
cx q[4],q[6];
ry(-2.186929187986416) q[1];
ry(-0.24964999019335457) q[3];
cx q[1],q[3];
ry(3.0937393394251314) q[1];
ry(-0.024753537885138194) q[3];
cx q[1],q[3];
ry(-1.7270832046103817) q[3];
ry(-1.9890160828888472) q[5];
cx q[3],q[5];
ry(0.008535191947451167) q[3];
ry(3.1219290281313454) q[5];
cx q[3],q[5];
ry(-1.4900497916439146) q[5];
ry(3.037946993210878) q[7];
cx q[5],q[7];
ry(-3.100492053886166) q[5];
ry(-3.1323969368384126) q[7];
cx q[5],q[7];
ry(2.542230260112239) q[0];
ry(1.6046653632407675) q[1];
ry(-0.6415547544364986) q[2];
ry(1.7080069569125047) q[3];
ry(2.3504615739691115) q[4];
ry(2.4623171302241693) q[5];
ry(-1.6109350328442362) q[6];
ry(2.992345266380329) q[7];