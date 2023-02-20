OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.669296232480311) q[0];
rz(-1.5720724490458908) q[0];
ry(1.5703619617174795) q[1];
rz(-0.8376111032455009) q[1];
ry(-1.571567555863229) q[2];
rz(1.5706052348370996) q[2];
ry(-1.5710579682019357) q[3];
rz(2.2002247030824122) q[3];
ry(-1.5709302782815726) q[4];
rz(-1.570860529689387) q[4];
ry(2.3155780514866415) q[5];
rz(1.2471344598062204) q[5];
ry(0.00022142945337577655) q[6];
rz(2.0019620882366045) q[6];
ry(3.141238313453838) q[7];
rz(-0.5221714563153579) q[7];
ry(1.5698278457624741) q[8];
rz(-1.5709229718551556) q[8];
ry(-1.57117043973071) q[9];
rz(0.003382528555846989) q[9];
ry(-1.4754711483098388) q[10];
rz(2.9577369415542143) q[10];
ry(-1.569778979677033) q[11];
rz(-2.2579494188659943) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5508243325644244) q[0];
rz(0.013629117900185506) q[0];
ry(1.677484328404123) q[1];
rz(0.007209099659495584) q[1];
ry(-1.56642348839355) q[2];
rz(-1.5085867669207313) q[2];
ry(0.0010731195402868068) q[3];
rz(-0.6299676087671346) q[3];
ry(2.237615621428718) q[4];
rz(2.3471411681770045) q[4];
ry(3.1408447836479563) q[5];
rz(0.8726476289498144) q[5];
ry(1.4330889491334942) q[6];
rz(0.0028860561744093885) q[6];
ry(1.558433009055273) q[7];
rz(-0.32307578491965455) q[7];
ry(1.5705813445681864) q[8];
rz(-2.2102772301129896) q[8];
ry(2.5337131557640133) q[9];
rz(-0.0013633561327464384) q[9];
ry(-2.7526849429509666) q[10];
rz(-2.803687796577676) q[10];
ry(-0.3117674496813265) q[11];
rz(-2.2369874322563064) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5712226908558768) q[0];
rz(-0.053530841451374656) q[0];
ry(-1.5712325785171157) q[1];
rz(-1.5539677540915058) q[1];
ry(3.1411964848195573) q[2];
rz(-3.079071599109126) q[2];
ry(1.5708357570313205) q[3];
rz(1.8830922427590266) q[3];
ry(-3.141392101881267) q[4];
rz(2.1979801205260534) q[4];
ry(3.141555121191469) q[5];
rz(2.2558885297619) q[5];
ry(-2.7389160139314956) q[6];
rz(3.0081740323138355) q[6];
ry(-1.8145956255202675) q[7];
rz(-2.929554068010243) q[7];
ry(3.0946943196219077) q[8];
rz(-2.157617455128211) q[8];
ry(-0.00034175515312817595) q[9];
rz(-3.1375553292128044) q[9];
ry(3.1401036473381625) q[10];
rz(0.5526914359133156) q[10];
ry(3.1412610942950816) q[11];
rz(1.6493593280358116) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.2058183901432176) q[0];
rz(-2.459655610715655) q[0];
ry(1.4146997173965565) q[1];
rz(-2.787262138909294) q[1];
ry(-1.570964245340052) q[2];
rz(0.7019020454178094) q[2];
ry(-0.0003849285137331388) q[3];
rz(2.8283331924013475) q[3];
ry(0.00013963512945736056) q[4];
rz(-1.4218045017680183) q[4];
ry(0.00016162261968740986) q[5];
rz(-1.157031669300214) q[5];
ry(-3.067202153413651) q[6];
rz(1.4676367446303429) q[6];
ry(3.128751170338048) q[7];
rz(-2.8894086164902357) q[7];
ry(-3.141466529626916) q[8];
rz(-1.8978758118468182) q[8];
ry(1.6748327286682754) q[9];
rz(0.15559851943841263) q[9];
ry(2.1249234194766835) q[10];
rz(-2.2949646448294683) q[10];
ry(2.0054022194509606) q[11];
rz(1.3052646267687877) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.0009272023100405298) q[0];
rz(-0.6808747030810381) q[0];
ry(-7.897021037812303e-05) q[1];
rz(-1.9286055586155566) q[1];
ry(-0.0007621693668529161) q[2];
rz(-0.7025523665749621) q[2];
ry(-0.05765772615684977) q[3];
rz(-1.5692658178541046) q[3];
ry(1.5704233268534438) q[4];
rz(1.19291500613223) q[4];
ry(3.141517066249444) q[5];
rz(-2.0915580521792094) q[5];
ry(-1.5643160645533056) q[6];
rz(-2.6487371271784137) q[6];
ry(1.8387263450763087) q[7];
rz(-1.4827099904041798) q[7];
ry(0.6248516193287915) q[8];
rz(1.620541191283014) q[8];
ry(-0.001886024915514817) q[9];
rz(-2.2258564579372737) q[9];
ry(0.0033053303386100883) q[10];
rz(0.8080358006840999) q[10];
ry(1.5660318076635127) q[11];
rz(1.5705807683664994) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.205374464142885) q[0];
rz(1.6181952833154716) q[0];
ry(-1.589752352492023) q[1];
rz(-1.4641552051046276) q[1];
ry(-1.5715148530320318) q[2];
rz(0.01719383957396925) q[2];
ry(1.5688420356212651) q[3];
rz(2.993624603727863) q[3];
ry(-3.1411424379161206) q[4];
rz(-2.4936812746187957) q[4];
ry(3.1415466399856093) q[5];
rz(1.2764749077373816) q[5];
ry(3.2595291315640145e-05) q[6];
rz(-2.315203303845592) q[6];
ry(-8.691987844527443e-05) q[7];
rz(0.5877185221390533) q[7];
ry(2.914091474056746) q[8];
rz(0.40839434217569615) q[8];
ry(0.00010204299029936692) q[9];
rz(2.0700897728115275) q[9];
ry(-1.5711295501549536) q[10];
rz(0.054059624733693745) q[10];
ry(1.6255804789475892) q[11];
rz(0.0011703817438890596) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.509422453457403) q[0];
rz(1.6709053376891376) q[0];
ry(-3.01585914186749) q[1];
rz(-2.8797088343771247) q[1];
ry(-0.022203228990952617) q[2];
rz(-0.6355839381664624) q[2];
ry(0.41966694342705857) q[3];
rz(0.7590988091581972) q[3];
ry(0.0031223959042154803) q[4];
rz(-2.687945242265289) q[4];
ry(-3.139157066180753) q[5];
rz(0.947979579197449) q[5];
ry(1.539565921316589) q[6];
rz(0.015534438712588753) q[6];
ry(0.5735382192393265) q[7];
rz(-0.6061165756090823) q[7];
ry(-1.5718898131913825) q[8];
rz(2.7402114445550785) q[8];
ry(-1.5606291860823642) q[9];
rz(-1.5705864789254227) q[9];
ry(1.9289969827659) q[10];
rz(0.07173093609519743) q[10];
ry(-1.5704104632174245) q[11];
rz(-0.3791657245666877) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.5716667629818786) q[0];
rz(1.5821747308632237) q[0];
ry(1.5674356544591472) q[1];
rz(-1.5902294761866398) q[1];
ry(0.0001927321612749111) q[2];
rz(0.6184076922761489) q[2];
ry(-3.141331776273196) q[3];
rz(1.1747920152743418) q[3];
ry(0.00019277299955806733) q[4];
rz(-0.083162240364926) q[4];
ry(0.00038903832463699217) q[5];
rz(-1.3504592743459298) q[5];
ry(-1.5679921381611326) q[6];
rz(-1.5701399238549367) q[6];
ry(-1.5729049503490595) q[7];
rz(-2.7248428400841718) q[7];
ry(3.1415459792659157) q[8];
rz(-0.4019704777187997) q[8];
ry(1.5709872895366477) q[9];
rz(0.08065870992630877) q[9];
ry(-3.141489044306381) q[10];
rz(0.7570314455826147) q[10];
ry(-0.00021690427681431146) q[11];
rz(-2.763428866576511) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-0.17772820689942304) q[0];
rz(-1.5817232709257898) q[0];
ry(-3.051448799476612) q[1];
rz(1.551562775309907) q[1];
ry(-1.5697655860858086) q[2];
rz(-0.9185329825150048) q[2];
ry(8.717827480211421e-05) q[3];
rz(-1.6961565232867954) q[3];
ry(-3.1401309986875097) q[4];
rz(2.0722850349436266) q[4];
ry(-0.0032300135315940343) q[5];
rz(-1.1899424472108195) q[5];
ry(1.5714734101613272) q[6];
rz(-1.270050118893911) q[6];
ry(-1.5702811212240473) q[7];
rz(-1.57064021485232) q[7];
ry(-1.5709052382091873) q[8];
rz(-0.0004859504392135165) q[8];
ry(0.0007714917375528892) q[9];
rz(-1.0318520755710852) q[9];
ry(3.140999315972295) q[10];
rz(-2.521922334426684) q[10];
ry(-1.5632870623138242) q[11];
rz(-1.5743570778442786) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5732529631807426) q[0];
rz(1.4996970138468906) q[0];
ry(1.5723340922674862) q[1];
rz(-3.1409999305213883) q[1];
ry(3.1411981065314016) q[2];
rz(2.2058756569380686) q[2];
ry(-3.14058946785003) q[3];
rz(2.022336013265709) q[3];
ry(3.1414400643173876) q[4];
rz(-0.5798101877461963) q[4];
ry(0.012349507696254276) q[5];
rz(0.2867681857629938) q[5];
ry(-0.0005536221014992128) q[6];
rz(0.6407325526312112) q[6];
ry(1.5695877855719598) q[7];
rz(-3.107810433215623) q[7];
ry(-1.5713553150211084) q[8];
rz(-1.4247620452352598) q[8];
ry(3.1324009294753186) q[9];
rz(2.194704420427544) q[9];
ry(3.1402834214811532) q[10];
rz(-2.202463280456432) q[10];
ry(1.571509859419912) q[11];
rz(-1.7199379026383064) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.1096627896412787) q[0];
rz(1.9555600839265663) q[0];
ry(-1.5709820732720592) q[1];
rz(-2.938922227552408) q[1];
ry(3.1194102718694854) q[2];
rz(-0.01029489295245212) q[2];
ry(-1.9853601790705413) q[3];
rz(1.6363542059179763) q[3];
ry(3.1415732412943345) q[4];
rz(0.3143828560688611) q[4];
ry(-6.753734908659936e-05) q[5];
rz(3.002103828701112) q[5];
ry(-0.0007268643116269402) q[6];
rz(-2.511869132195126) q[6];
ry(3.141282872499721) q[7];
rz(2.951072580497075) q[7];
ry(0.0009717030310226) q[8];
rz(1.1760688215376074) q[8];
ry(-2.731713444446194) q[9];
rz(2.97871981653895) q[9];
ry(3.1300847621477725) q[10];
rz(2.567769676814687) q[10];
ry(-0.13489490260222414) q[11];
rz(-1.4263951763248377) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.1413505408072795) q[0];
rz(0.45557720324675754) q[0];
ry(3.1411820201076157) q[1];
rz(0.799174493822154) q[1];
ry(1.5733209579905483) q[2];
rz(5.364811121077995e-05) q[2];
ry(-1.5726416582546767) q[3];
rz(1.5707525621062768) q[3];
ry(-1.5706343816605444) q[4];
rz(8.182432971892695e-05) q[4];
ry(1.5756987380994445) q[5];
rz(-3.140662892329393) q[5];
ry(-1.5690384032412201) q[6];
rz(-1.8228072091688254) q[6];
ry(0.0013254654713747982) q[7];
rz(1.5228118818382397) q[7];
ry(3.14075070303675) q[8];
rz(1.3258070825380637) q[8];
ry(-3.085164113316472) q[9];
rz(0.7397390379920842) q[9];
ry(1.5658207679094067) q[10];
rz(1.5081813015398566) q[10];
ry(1.5721338724247782) q[11];
rz(1.5706482629441587) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(1.570429845727154) q[0];
rz(-3.1414151572727036) q[0];
ry(-3.1412075209646146) q[1];
rz(-1.1129039708155806) q[1];
ry(1.5709434354313754) q[2];
rz(-0.571731467655626) q[2];
ry(-1.5709377356799425) q[3];
rz(-1.571301344884331) q[3];
ry(-1.5707902947221537) q[4];
rz(-0.00010035475630913395) q[4];
ry(-1.57076064292192) q[5];
rz(-3.1407957542060942) q[5];
ry(3.141571844871784) q[6];
rz(0.9185445165757402) q[6];
ry(-6.741875082472149e-05) q[7];
rz(-1.2981161324544654) q[7];
ry(0.1876279476148639) q[8];
rz(-2.911910807323502) q[8];
ry(3.1415095429638775) q[9];
rz(-1.9184373727112964) q[9];
ry(-3.1394834123678934) q[10];
rz(-1.6340425248260784) q[10];
ry(1.5773193676116657) q[11];
rz(-0.16137081342679405) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5705092385658357) q[0];
rz(2.030238367503496) q[0];
ry(0.08486028422191882) q[1];
rz(0.9265473007522737) q[1];
ry(0.00014767738545777565) q[2];
rz(-2.6141531776973466) q[2];
ry(-1.5709156368092971) q[3];
rz(-3.1415030420219954) q[3];
ry(1.5708191608227864) q[4];
rz(-1.5703687416015875) q[4];
ry(-1.5706136869205158) q[5];
rz(2.9466445417999023) q[5];
ry(-0.00032503279319717065) q[6];
rz(-1.1699102945473046) q[6];
ry(1.571155458679962) q[7];
rz(1.4469504762899297) q[7];
ry(3.1413110905714685) q[8];
rz(-1.213395276441956) q[8];
ry(0.00013709182062093814) q[9];
rz(-0.3171904846388401) q[9];
ry(-1.5705445916806957) q[10];
rz(-2.8009726920967872) q[10];
ry(3.128575885121927) q[11];
rz(1.4086538577270618) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.1402807630858383) q[0];
rz(2.212134022368784) q[0];
ry(-0.0005195832412775501) q[1];
rz(2.414558265041129) q[1];
ry(-0.00020160458602447306) q[2];
rz(-2.91609412851979) q[2];
ry(1.5690893945441038) q[3];
rz(0.20537626620521277) q[3];
ry(-1.570842688753187) q[4];
rz(-1.3904439120599037) q[4];
ry(3.1414692754286) q[5];
rz(-1.561066657669513) q[5];
ry(1.5706186774452036) q[6];
rz(0.18000110252255636) q[6];
ry(-0.0005553955755437889) q[7];
rz(1.8970248423766156) q[7];
ry(0.00026550115470325153) q[8];
rz(-1.5167470965692407) q[8];
ry(1.569095004581816) q[9];
rz(1.7734076460017425) q[9];
ry(-0.00034053208065871495) q[10];
rz(1.4001529149353553) q[10];
ry(-1.588157504980732) q[11];
rz(-2.9431481183457486) q[11];