OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.14148008585431) q[0];
rz(0.05142074734001617) q[0];
ry(3.1414406140286224) q[1];
rz(-2.229674028632351) q[1];
ry(3.1414427548727955) q[2];
rz(1.173701985389374) q[2];
ry(-3.1414011647247144) q[3];
rz(-2.950136008407857) q[3];
ry(-0.0001422679663123657) q[4];
rz(-0.9087826727046457) q[4];
ry(0.0005630148801447632) q[5];
rz(-0.6864392750110778) q[5];
ry(3.1415722104179884) q[6];
rz(1.0373735646452524) q[6];
ry(-3.1305708250707953) q[7];
rz(-3.124521277011056) q[7];
ry(-1.79466125735763) q[8];
rz(1.5705466240540613) q[8];
ry(-1.5712627755031159) q[9];
rz(-1.5695621004874423) q[9];
ry(0.10353768355314742) q[10];
rz(-0.00034810343648761233) q[10];
ry(-3.141505419979829) q[11];
rz(1.3959703602936895) q[11];
ry(2.897683149240581) q[12];
rz(3.1415274424607356) q[12];
ry(2.0938583767033716) q[13];
rz(-3.1415871554966768) q[13];
ry(1.5708119712920563) q[14];
rz(0.008913839515663824) q[14];
ry(-2.6770626331164875) q[15];
rz(-3.997715130399371e-05) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.137678870757329) q[0];
rz(-2.3453246755985435) q[0];
ry(-3.1332944362459703) q[1];
rz(-2.934269991379773) q[1];
ry(3.1310439617531345) q[2];
rz(0.003074001225035648) q[2];
ry(3.1349881560540878) q[3];
rz(-3.1140776180173613) q[3];
ry(0.00330347939861109) q[4];
rz(-0.0781598476988874) q[4];
ry(-0.014992780973583919) q[5];
rz(3.1379410477950453) q[5];
ry(3.1406231339874306) q[6];
rz(-0.3714007834540665) q[6];
ry(3.080001904190739) q[7];
rz(-0.0013906509637006733) q[7];
ry(1.5706467513833589) q[8];
rz(-3.133370457304233) q[8];
ry(1.5714791954218938) q[9];
rz(2.6426907389861736) q[9];
ry(2.50561493030342) q[10];
rz(0.0010003589403630995) q[10];
ry(1.5707985922091146) q[11];
rz(0.0006220798829463803) q[11];
ry(-2.1994183529996643) q[12];
rz(-3.140700750193184) q[12];
ry(-2.6550533666717593) q[13];
rz(-0.18283907033068528) q[13];
ry(-1.2268962591428891) q[14];
rz(3.0246307575008737) q[14];
ry(0.11214405628004025) q[15];
rz(0.33375298454764923) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.1411527011100437) q[0];
rz(-0.20201902585961345) q[0];
ry(-0.0022179580355903283) q[1];
rz(2.9192719400472695) q[1];
ry(3.1354800093828654) q[2];
rz(0.23660113937415653) q[2];
ry(-3.0933738606910053) q[3];
rz(1.671639830593814) q[3];
ry(0.23425616228084203) q[4];
rz(0.00011799492324964227) q[4];
ry(-2.21338990746099) q[5];
rz(-0.0002015270389827961) q[5];
ry(1.544025416122979) q[6];
rz(-3.1415738805441027) q[6];
ry(2.695309947844556) q[7];
rz(-0.0009633964604116585) q[7];
ry(3.0809888888462553) q[8];
rz(1.571263517988619) q[8];
ry(-6.836527994558139e-05) q[9];
rz(0.8083339366216205) q[9];
ry(0.05344479895567033) q[10];
rz(-3.1404187585832237) q[10];
ry(-1.5179777622529196) q[11];
rz(-1.5714432997031609) q[11];
ry(3.117297970103568) q[12];
rz(-0.2601538720607906) q[12];
ry(-3.1403798521739166) q[13];
rz(1.1314910247970436) q[13];
ry(3.1415794802351855) q[14];
rz(3.0288844124168546) q[14];
ry(-0.0004997288327137403) q[15];
rz(2.6940130332730154) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.0003425190115004934) q[0];
rz(-0.6499753161619739) q[0];
ry(-3.1403271122491376) q[1];
rz(-2.3322462536212596) q[1];
ry(0.0011471825827509457) q[2];
rz(-0.8478186558751427) q[2];
ry(3.1414597801556208) q[3];
rz(-2.3213931937618515) q[3];
ry(3.128606482053614) q[4];
rz(3.1098713312222364) q[4];
ry(-3.0642499859621064) q[5];
rz(-1.2235563772674158) q[5];
ry(1.5280970495248) q[6];
rz(-1.459757551757784) q[6];
ry(2.9243735596860447) q[7];
rz(-1.7434034381954042) q[7];
ry(-0.005425597079855571) q[8];
rz(-1.5525283909074288) q[8];
ry(3.14077547391958) q[9];
rz(-2.967043937080748) q[9];
ry(2.9220948771962747) q[10];
rz(-3.0832770881312026) q[10];
ry(1.570196309299571) q[11];
rz(1.5882376458438658) q[11];
ry(-3.1385230145723617) q[12];
rz(1.306970117504695) q[12];
ry(0.00010879827373422302) q[13];
rz(-2.8962712500270316) q[13];
ry(1.5584880434578774) q[14];
rz(-1.7289967422064019) q[14];
ry(0.001797995523640239) q[15];
rz(-1.4774755930310646) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-5.7412684855129914e-05) q[0];
rz(0.04135019449521697) q[0];
ry(-3.1415207140282346) q[1];
rz(-0.7333553303911099) q[1];
ry(-3.141542116089031) q[2];
rz(-2.1900155861840553) q[2];
ry(-3.141530819967495) q[3];
rz(0.7233198469282511) q[3];
ry(-3.1415706660312614) q[4];
rz(-1.6028454501205005) q[4];
ry(3.1412054502016797) q[5];
rz(0.3378569010652708) q[5];
ry(1.671410615688984e-05) q[6];
rz(-0.06569308139741827) q[6];
ry(0.0008958165771710279) q[7];
rz(-3.0637521363966136) q[7];
ry(-1.5706898929781294) q[8];
rz(-1.2704833824720934) q[8];
ry(-1.5700591325810151) q[9];
rz(3.1131930745840775) q[9];
ry(-0.00010120237909028162) q[10];
rz(-1.664975384658879) q[10];
ry(8.622668312841818e-05) q[11];
rz(1.5530692427449215) q[11];
ry(-3.134448813259811) q[12];
rz(-3.0553241470915578) q[12];
ry(-0.006071228421422603) q[13];
rz(-2.171928926798643) q[13];
ry(-3.0889470265639827) q[14];
rz(1.413481633682045) q[14];
ry(3.1380960642112665) q[15];
rz(1.3170113653502564) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-2.817228280135919) q[0];
rz(3.141569848037123) q[0];
ry(-2.239508931770985) q[1];
rz(3.141554034397821) q[1];
ry(2.047341879962207) q[2];
rz(-3.141520436367875) q[2];
ry(-0.6308149304906703) q[3];
rz(3.1415412171773047) q[3];
ry(-2.91229906237206) q[4];
rz(3.141545859328584) q[4];
ry(-3.112026767098234) q[5];
rz(3.131857082993564) q[5];
ry(-3.1312429252223213) q[6];
rz(0.04512333908721633) q[6];
ry(-3.1374266560276522) q[7];
rz(-0.09326417785237469) q[7];
ry(3.1332934684937874) q[8];
rz(-2.8063200566208084) q[8];
ry(1.5729828700322503) q[9];
rz(-1.6094210925165144) q[9];
ry(0.004486432608029389) q[10];
rz(-3.104411468595627) q[10];
ry(-1.5619951362112332) q[11];
rz(3.10486460743901) q[11];
ry(-3.138945300285387) q[12];
rz(0.22348423800288764) q[12];
ry(-3.1409508400807145) q[13];
rz(-2.282048870049378) q[13];
ry(1.570220297726804) q[14];
rz(1.6702282272842695) q[14];
ry(-8.325598696856673e-05) q[15];
rz(-1.5851545877987894) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(2.8387924075192785) q[0];
rz(1.0507557561964587) q[0];
ry(0.8257665685790458) q[1];
rz(1.0507877144961801) q[1];
ry(-0.5038768852573705) q[2];
rz(1.0507114633149286) q[2];
ry(-0.7889684057802511) q[3];
rz(-2.090789978681849) q[3];
ry(0.732868322532664) q[4];
rz(-2.0907054772365843) q[4];
ry(2.2578546836398186) q[5];
rz(-2.090608495851156) q[5];
ry(2.041106866137216) q[6];
rz(-2.0905599916216664) q[6];
ry(0.3376502887555973) q[7];
rz(1.0506232762421919) q[7];
ry(-0.012547096674365297) q[8];
rz(-2.1243765161109556) q[8];
ry(-3.133943982664001) q[9];
rz(1.0190280075967424) q[9];
ry(-3.1373866891014384) q[10];
rz(-2.08919029010807) q[10];
ry(-0.008607344846597725) q[11];
rz(1.0901237583221537) q[11];
ry(3.1380422795133085) q[12];
rz(1.187815690628889) q[12];
ry(-3.1365002938652196) q[13];
rz(0.9544142218597045) q[13];
ry(0.007461371579224804) q[14];
rz(0.9539600783059015) q[14];
ry(3.139401708293272) q[15];
rz(-2.3357886710094253) q[15];