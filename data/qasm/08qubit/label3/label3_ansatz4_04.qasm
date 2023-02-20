OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.056632790742064495) q[0];
rz(0.04876666769296633) q[0];
ry(1.589560297045412) q[1];
rz(1.5716639023484404) q[1];
ry(3.077765010074084) q[2];
rz(-1.6831149807251977) q[2];
ry(-3.308856033612528e-07) q[3];
rz(-1.1003204579982944) q[3];
ry(1.5697935411239452) q[4];
rz(2.0705572282559004) q[4];
ry(6.8655629359426484e-06) q[5];
rz(0.2683554089789242) q[5];
ry(-1.598500875613718) q[6];
rz(-1.5734790850130282) q[6];
ry(-0.07099916383564589) q[7];
rz(1.5755397737959114) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.174334152341364) q[0];
rz(1.5707723617305442) q[0];
ry(1.6195139758326622) q[1];
rz(0.332395210516463) q[1];
ry(0.002653022962451068) q[2];
rz(1.1075386668711067) q[2];
ry(1.5707955293442633) q[3];
rz(1.5707891678582993) q[3];
ry(-1.570708716434984) q[4];
rz(-2.243004039197736) q[4];
ry(3.141569719691136) q[5];
rz(1.9537194606243022) q[5];
ry(-0.34867620238555364) q[6];
rz(0.0018576201801652203) q[6];
ry(0.6317131629535959) q[7];
rz(-3.138517259427195) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5707965237237542) q[0];
rz(-0.4551753446750837) q[0];
ry(-1.5704867384256005) q[1];
rz(1.5708517707862315) q[1];
ry(-4.119235565092083e-06) q[2];
rz(2.091265402877511) q[2];
ry(3.1096585124923255) q[3];
rz(0.0002213185047156344) q[3];
ry(-3.141591150400805) q[4];
rz(0.8982346131617377) q[4];
ry(-2.7007971109899285) q[5];
rz(-1.2603413968053) q[5];
ry(1.4826612272673527) q[6];
rz(-3.1363090941503455) q[6];
ry(1.6620955673633455) q[7];
rz(1.9285290315739632) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.14159137428049) q[0];
rz(1.1163764946086077) q[0];
ry(-1.0056760770355007) q[1];
rz(3.1415412930054867) q[1];
ry(1.570791413388623) q[2];
rz(-2.3180894851555465) q[2];
ry(-3.1123232102843446) q[3];
rz(7.429531929344084e-05) q[3];
ry(-1.5709253280255442) q[4];
rz(-3.1266655727530024) q[4];
ry(-9.388253396627671e-06) q[5];
rz(-1.8415597842006854) q[5];
ry(-0.9389264143375877) q[6];
rz(-1.5692869617557186) q[6];
ry(-3.1409425176187526) q[7];
rz(0.3582105850408297) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.69250725662408) q[0];
rz(0.0005916505595067179) q[0];
ry(1.5708011082657654) q[1];
rz(-2.0643915952405074e-05) q[1];
ry(3.553716842328032e-08) q[2];
rz(-0.5504519970158137) q[2];
ry(2.904598403690832) q[3];
rz(-1.50764964858935) q[3];
ry(1.5707897696003175) q[4];
rz(-3.1379984328931423) q[4];
ry(-0.0019716441951044355) q[5];
rz(3.10398161171956) q[5];
ry(-1.4254560708630137) q[6];
rz(0.0003910674331140576) q[6];
ry(-1.483828913194567) q[7];
rz(0.6226996736108594) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4122018079045369) q[0];
rz(1.5705267257340154) q[0];
ry(-2.9940179582213484) q[1];
rz(-0.48231844456765277) q[1];
ry(-1.5707944668212477) q[2];
rz(1.5709963815628902) q[2];
ry(-3.13714804990007) q[3];
rz(-0.12646232258895562) q[3];
ry(-0.00035031855963474356) q[4];
rz(1.1377335213581883) q[4];
ry(1.569602410498641) q[5];
rz(1.5707748135688626) q[5];
ry(-1.570833795933342) q[6];
rz(1.6800490973649156) q[6];
ry(-3.1415868780201515) q[7];
rz(2.1963144251223605) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5707561185877514) q[0];
rz(-0.35411457944018654) q[0];
ry(-3.1415836863511983) q[1];
rz(0.5059599483574532) q[1];
ry(0.008963510570230744) q[2];
rz(0.29336115877323815) q[2];
ry(6.039428766868582e-05) q[3];
rz(-2.95235754433605) q[3];
ry(3.141588193770341) q[4];
rz(-2.1336255883724844) q[4];
ry(-2.832489568976029) q[5];
rz(-3.141337001773407) q[5];
ry(-1.1278224903780245e-05) q[6];
rz(2.311074309154137) q[6];
ry(-1.5707954825608945) q[7];
rz(-1.5704247189160405) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.00011034375509347457) q[0];
rz(-1.786124208658481) q[0];
ry(0.0005006089443195805) q[1];
rz(-1.5576907888783822) q[1];
ry(3.1415917916445557) q[2];
rz(-0.27588178857787177) q[2];
ry(-1.5705044997353845) q[3];
rz(-0.5694401899469183) q[3];
ry(-3.141565738247947) q[4];
rz(-2.2736101137586333) q[4];
ry(-1.570419404041951) q[5];
rz(-2.1402344182421853) q[5];
ry(3.141564263388601) q[6];
rz(0.2800784109179446) q[6];
ry(-1.5710593181159123) q[7];
rz(2.572160868300535) q[7];