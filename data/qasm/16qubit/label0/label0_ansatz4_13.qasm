OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.1282510655813267) q[0];
rz(0.42547162970079727) q[0];
ry(3.1403221764135294) q[1];
rz(0.16344076937739604) q[1];
ry(1.5735055192874263) q[2];
rz(0.37497128972310634) q[2];
ry(1.5680812296027598) q[3];
rz(-0.5485253134473691) q[3];
ry(-3.1395386673227392) q[4];
rz(-2.405534130177574) q[4];
ry(-3.140618938917052) q[5];
rz(-1.2055219114538216) q[5];
ry(-0.5674054400082875) q[6];
rz(-0.8169859660864622) q[6];
ry(-2.903565911643823) q[7];
rz(-2.965361781886696) q[7];
ry(0.0002077394118149556) q[8];
rz(-0.24361397847076652) q[8];
ry(3.1410133773361135) q[9];
rz(0.6387615508121423) q[9];
ry(1.5810277505817236) q[10];
rz(2.701924784428934) q[10];
ry(1.5611176582938313) q[11];
rz(0.06599507366057276) q[11];
ry(-3.1411105795332515) q[12];
rz(-2.063943258470789) q[12];
ry(3.323144377365139e-05) q[13];
rz(-0.5569368449968886) q[13];
ry(1.9207270834782122) q[14];
rz(0.5026922131924163) q[14];
ry(1.0332379002632819) q[15];
rz(-2.9928632557391985) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.0965885139015397) q[0];
rz(-0.7215869121406945) q[0];
ry(-0.006578196746368725) q[1];
rz(0.8948432466394695) q[1];
ry(-2.123204398669987) q[2];
rz(2.5459543401435516) q[2];
ry(-2.5183458483965437) q[3];
rz(-2.059403071540917) q[3];
ry(2.7675172268010897) q[4];
rz(-1.2046951530164636) q[4];
ry(-1.8818050600734246) q[5];
rz(0.043394618009624665) q[5];
ry(0.1894346343818043) q[6];
rz(-2.0014910074499714) q[6];
ry(2.9325031644699076) q[7];
rz(0.8067958319344833) q[7];
ry(1.677294391713784) q[8];
rz(-1.4072925590094618) q[8];
ry(-3.0390386626363024) q[9];
rz(2.3723021196699734) q[9];
ry(1.7715940660521463) q[10];
rz(-2.9460778820197886) q[10];
ry(-0.6115881709425857) q[11];
rz(-1.279997812367359) q[11];
ry(-0.13955614234605918) q[12];
rz(-1.4319270135210886) q[12];
ry(-0.15806423431297123) q[13];
rz(-2.583269748470945) q[13];
ry(-0.8829694240140417) q[14];
rz(-0.8706449083759108) q[14];
ry(-2.1206174202534536) q[15];
rz(0.6640181351574479) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(3.1309350297385796) q[0];
rz(-0.47913704388231865) q[0];
ry(-0.008466368630617893) q[1];
rz(2.906981892084824) q[1];
ry(3.1134413045697094) q[2];
rz(-0.934132732733169) q[2];
ry(-2.8605016571384754) q[3];
rz(-0.4125161320767887) q[3];
ry(-3.078486124087404) q[4];
rz(-1.8620484294524078) q[4];
ry(-0.08282402380861774) q[5];
rz(1.8825307799966646) q[5];
ry(0.0005948203331858082) q[6];
rz(-2.483385596474472) q[6];
ry(-3.140800372631369) q[7];
rz(-1.1298374939569173) q[7];
ry(0.001881648426405036) q[8];
rz(-3.0838375275369008) q[8];
ry(3.141276406804101) q[9];
rz(-1.5455737167851078) q[9];
ry(-0.00634219978773692) q[10];
rz(0.19322637575946078) q[10];
ry(-3.1404886190245325) q[11];
rz(1.8376447945390588) q[11];
ry(-3.141558756070531) q[12];
rz(-1.89723733768335) q[12];
ry(-0.002118614009286368) q[13];
rz(-0.7469986780443056) q[13];
ry(0.2353397782296496) q[14];
rz(-0.9745318364738669) q[14];
ry(0.4769412307654036) q[15];
rz(0.4623316182502411) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.20334823018983883) q[0];
rz(2.859062788158276) q[0];
ry(0.039407926551303187) q[1];
rz(-2.377785775701047) q[1];
ry(1.3081465118102122) q[2];
rz(-0.7713809485175152) q[2];
ry(-3.0997495688210703) q[3];
rz(1.808622772426536) q[3];
ry(-3.1386012303858295) q[4];
rz(2.52395815020993) q[4];
ry(3.127756402430643) q[5];
rz(2.387535234127601) q[5];
ry(-0.37541801370150285) q[6];
rz(0.003838493488603767) q[6];
ry(3.1134552074676867) q[7];
rz(0.8171205830430333) q[7];
ry(-2.119116760485496) q[8];
rz(-1.7307826816413447) q[8];
ry(1.6420866635611082) q[9];
rz(-0.49174662784307765) q[9];
ry(-2.4804513327948365) q[10];
rz(1.7565908412417819) q[10];
ry(1.7305749513450266) q[11];
rz(-1.851643790313421) q[11];
ry(-2.138783945679517) q[12];
rz(0.2095094473258694) q[12];
ry(0.029160996714264975) q[13];
rz(-1.0492228673638382) q[13];
ry(-2.9762623582722147) q[14];
rz(-1.8695548276125808) q[14];
ry(-0.3484159023173703) q[15];
rz(2.1316984230590084) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.5941025522965893) q[0];
rz(1.5995075101600884) q[0];
ry(-3.1277833192276585) q[1];
rz(-2.924426259388779) q[1];
ry(-3.119492565089907) q[2];
rz(-1.2652266257574025) q[2];
ry(0.008824831521124565) q[3];
rz(-2.7515837065825166) q[3];
ry(2.6309772579578046) q[4];
rz(-2.1624152300687456) q[4];
ry(-1.7214795833247507) q[5];
rz(1.265186326304618) q[5];
ry(-0.22218999830514344) q[6];
rz(-2.527766193906215) q[6];
ry(0.9951948034802031) q[7];
rz(0.950612834181309) q[7];
ry(-3.1232894229933343) q[8];
rz(-1.8004264251536535) q[8];
ry(-3.1398851134437864) q[9];
rz(-1.9936962490061654) q[9];
ry(0.36387803849508904) q[10];
rz(1.4541911747481804) q[10];
ry(0.35883596233120624) q[11];
rz(2.1809524608910547) q[11];
ry(0.7051449566347854) q[12];
rz(-1.7322962601305614) q[12];
ry(3.0291169940781844) q[13];
rz(0.3644472660656817) q[13];
ry(1.5447246857606176) q[14];
rz(1.9401713806119731) q[14];
ry(2.961598272967527) q[15];
rz(-2.6043540752612144) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.800732583826508) q[0];
rz(1.7699683034546387) q[0];
ry(-3.1179474183116906) q[1];
rz(1.2155226181409182) q[1];
ry(0.009142190622248605) q[2];
rz(2.104075120846006) q[2];
ry(-3.9051339580709055e-05) q[3];
rz(-0.8490062539946441) q[3];
ry(0.00016516451535242285) q[4];
rz(-1.1319812967507519) q[4];
ry(-0.00012797285745413944) q[5];
rz(1.614997880188449) q[5];
ry(3.1414537506532265) q[6];
rz(-0.014619474716294082) q[6];
ry(-3.137067023579695) q[7];
rz(2.7280054014857154) q[7];
ry(1.1372493691645849) q[8];
rz(-1.9948264516426153) q[8];
ry(0.6073108988720506) q[9];
rz(-0.7426807154173938) q[9];
ry(3.1406004739708284) q[10];
rz(1.3095480688186136) q[10];
ry(-3.1414694397497436) q[11];
rz(-0.9387493516640495) q[11];
ry(0.008130732086172014) q[12];
rz(-2.538772789863072) q[12];
ry(0.009966568587107455) q[13];
rz(-3.0131010009737333) q[13];
ry(2.328715144260099) q[14];
rz(-1.2864820943642368) q[14];
ry(-1.8141024730051336) q[15];
rz(1.4674877372222785) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5202401097682214) q[0];
rz(-1.4211576446487166) q[0];
ry(0.04662272318224403) q[1];
rz(-1.4787563424494614) q[1];
ry(-1.594054510922452) q[2];
rz(-0.06180826265751005) q[2];
ry(1.5821134672450672) q[3];
rz(-1.4508717815029417) q[3];
ry(-1.7500801488326678) q[4];
rz(-1.1045841729428008) q[4];
ry(1.1619826700847993) q[5];
rz(-0.09501450839509518) q[5];
ry(3.1152658568273828) q[6];
rz(2.004637969514354) q[6];
ry(0.9505639633001871) q[7];
rz(-0.20174952914582253) q[7];
ry(3.1322109629755253) q[8];
rz(-0.31380868048398014) q[8];
ry(3.1309882515423624) q[9];
rz(-2.5410957340990867) q[9];
ry(1.5906474740527372) q[10];
rz(-0.26364642313127834) q[10];
ry(-1.5873648874455348) q[11];
rz(-0.1419991669759359) q[11];
ry(-0.6317746568100472) q[12];
rz(1.762984208543755) q[12];
ry(-0.052845408488465744) q[13];
rz(2.263905853974008) q[13];
ry(-1.027359869400784) q[14];
rz(0.12036761212004818) q[14];
ry(-1.7461460508890898) q[15];
rz(1.7763508127523475) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.8706566429425163) q[0];
rz(-0.12752563193219532) q[0];
ry(-0.01069310307117671) q[1];
rz(-2.7926503904129456) q[1];
ry(3.139377280574675) q[2];
rz(-2.6127265067412355) q[2];
ry(2.9323472303384914) q[3];
rz(1.4645289733806048) q[3];
ry(0.0006063118621630892) q[4];
rz(1.354241923363211) q[4];
ry(-3.1413534246458905) q[5];
rz(-0.8818039957863919) q[5];
ry(-0.006945047829617579) q[6];
rz(-2.831350212640644) q[6];
ry(-0.03608102320891593) q[7];
rz(-0.9781346038322511) q[7];
ry(-1.4581950470438718) q[8];
rz(0.8213854885139873) q[8];
ry(-0.00018641246141059328) q[9];
rz(0.03605603916586287) q[9];
ry(-3.1074743633026514) q[10];
rz(-2.4993738396050107) q[10];
ry(3.014518533654506) q[11];
rz(1.4875247012522799) q[11];
ry(-1.5291065310021636) q[12];
rz(-1.3665410212214641) q[12];
ry(3.076635490767885) q[13];
rz(2.6040145272743813) q[13];
ry(-2.4326690157540356) q[14];
rz(2.9344866414374002) q[14];
ry(-1.222774008011914) q[15];
rz(-1.2632768200498565) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-3.1347882106642313) q[0];
rz(3.0064220908217822) q[0];
ry(-3.1398528442537725) q[1];
rz(0.5789401995900434) q[1];
ry(3.141232205027754) q[2];
rz(-2.815017007094932) q[2];
ry(-3.131990133645441) q[3];
rz(0.7889917154563904) q[3];
ry(2.134299571727153) q[4];
rz(-2.4031574393302924) q[4];
ry(0.5240299411840423) q[5];
rz(-2.3058835734088055) q[5];
ry(-1.7496422803269134) q[6];
rz(2.2890821534853565) q[6];
ry(1.3790305224215766) q[7];
rz(0.7956648542969996) q[7];
ry(0.020492547766512104) q[8];
rz(2.908095374607107) q[8];
ry(3.1393676933535746) q[9];
rz(1.41304717813206) q[9];
ry(0.009528647647925936) q[10];
rz(-2.3682482720191365) q[10];
ry(-0.06556851673749373) q[11];
rz(-1.5861866966560383) q[11];
ry(0.01503292304804943) q[12];
rz(0.08617360840146457) q[12];
ry(0.057527570922188485) q[13];
rz(-1.2442477878387503) q[13];
ry(2.581037840338249) q[14];
rz(3.133971811198753) q[14];
ry(-1.554801038353944) q[15];
rz(-2.3890185614325135) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.8718324459273674) q[0];
rz(0.2452483419264695) q[0];
ry(0.011183141573836151) q[1];
rz(-2.0444313772888902) q[1];
ry(3.1268162955212864) q[2];
rz(2.8240237231351504) q[2];
ry(2.5609097967621763) q[3];
rz(2.4580474338384315) q[3];
ry(-3.529783048195528e-05) q[4];
rz(-2.7305325995496745) q[4];
ry(-1.7200603602900194e-05) q[5];
rz(2.424547139365731) q[5];
ry(0.037072255037119015) q[6];
rz(0.20584982341619434) q[6];
ry(0.029236819528676072) q[7];
rz(-2.1812731575570563) q[7];
ry(-0.004323599031401168) q[8];
rz(-0.6695080462876835) q[8];
ry(-0.042498669407856446) q[9];
rz(2.1232003952306284) q[9];
ry(1.603363915573527) q[10];
rz(0.02912869059425205) q[10];
ry(1.4480024251045531) q[11];
rz(-1.5192989489373396) q[11];
ry(0.008127550978174213) q[12];
rz(2.447812289784681) q[12];
ry(-3.126481886428301) q[13];
rz(-0.06512051433258836) q[13];
ry(1.1478028409111636) q[14];
rz(-2.184143505382571) q[14];
ry(-2.437016408072618) q[15];
rz(-0.16608690032036225) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.527358445501578) q[0];
rz(-3.1199601846222778) q[0];
ry(-0.2582439316392691) q[1];
rz(-2.7290886901900917) q[1];
ry(3.1062197845131823) q[2];
rz(-2.5538560056082993) q[2];
ry(-1.5720158913275462) q[3];
rz(-1.3597190425390993) q[3];
ry(3.0598271582123298) q[4];
rz(2.793308237338368) q[4];
ry(-0.2774770426053425) q[5];
rz(0.9791035826643775) q[5];
ry(-0.18763571074143126) q[6];
rz(-0.9529547194035812) q[6];
ry(0.597684252121377) q[7];
rz(2.8205686019232354) q[7];
ry(0.00016761919382624058) q[8];
rz(2.093841248555367) q[8];
ry(-3.1407947071749764) q[9];
rz(2.252146788870865) q[9];
ry(1.5838367216475255) q[10];
rz(-0.1882031243314213) q[10];
ry(-3.1331835100118814) q[11];
rz(-1.5351814010748788) q[11];
ry(2.889778343203368) q[12];
rz(1.367653823378701) q[12];
ry(-3.1067150583962464) q[13];
rz(1.3283434492543766) q[13];
ry(-1.9046870786202865) q[14];
rz(-2.1824928461470723) q[14];
ry(3.055457141874437) q[15];
rz(0.7008570525619452) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(0.737061666805692) q[0];
rz(1.5677029460763747) q[0];
ry(0.5673576965026055) q[1];
rz(-2.0470393428711953) q[1];
ry(2.71852982691484) q[2];
rz(-2.2849865604750224) q[2];
ry(-1.2789712499681842) q[3];
rz(-1.564533929025342) q[3];
ry(-3.141081668593783) q[4];
rz(2.5698594332924243) q[4];
ry(-0.004123414117997702) q[5];
rz(2.858576534897042) q[5];
ry(1.6652503113897157) q[6];
rz(2.212072956143494) q[6];
ry(1.5539343788512214) q[7];
rz(2.139997775229576) q[7];
ry(1.569861122731647) q[8];
rz(-3.1412588904931598) q[8];
ry(1.5691047246787575) q[9];
rz(-0.0030729253369159433) q[9];
ry(-1.5948546447938083) q[10];
rz(-2.385268670834683) q[10];
ry(-0.357711094820123) q[11];
rz(2.0372786860823444) q[11];
ry(1.5707163439164578) q[12];
rz(-1.5704270154424795) q[12];
ry(1.5700231533615208) q[13];
rz(1.959550432834124) q[13];
ry(-1.5949740580347014) q[14];
rz(1.8786125609251787) q[14];
ry(1.5964534696399468) q[15];
rz(2.5267495017629478) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.5675949599732384) q[0];
rz(-3.0843464002802534) q[0];
ry(1.5760836502214932) q[1];
rz(-1.558379798442855) q[1];
ry(3.1393263860515632) q[2];
rz(-2.973875560998275) q[2];
ry(0.0003952363485009158) q[3];
rz(1.5972163108648427) q[3];
ry(-3.141288595308256) q[4];
rz(-2.167672550775433) q[4];
ry(3.1415609919545293) q[5];
rz(-2.538742798991584) q[5];
ry(-0.0029729569005834122) q[6];
rz(-2.193896669968431) q[6];
ry(-3.1410942221052354) q[7];
rz(2.149776552272423) q[7];
ry(1.5705663977055204) q[8];
rz(-1.571780687042713) q[8];
ry(1.5644583433979795) q[9];
rz(-1.5691375040072986) q[9];
ry(-0.0009553071608117136) q[10];
rz(-1.3757584864139505) q[10];
ry(0.000903235657910619) q[11];
rz(-2.756379739327647) q[11];
ry(-1.1322057552494087) q[12];
rz(1.569676567405745) q[12];
ry(-0.0019645506828931403) q[13];
rz(-1.9602969276919768) q[13];
ry(0.04772507740027976) q[14];
rz(-1.8757787308061653) q[14];
ry(0.028303242417353935) q[15];
rz(0.6215245591815969) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-1.476474346269165) q[0];
rz(2.5374415791452183) q[0];
ry(0.7555847019610722) q[1];
rz(-0.8289434841861913) q[1];
ry(1.55127171820018) q[2];
rz(-1.5699196186602107) q[2];
ry(0.8284516484658332) q[3];
rz(1.7233504418573522) q[3];
ry(3.138715058993703) q[4];
rz(0.07627097505623211) q[4];
ry(3.139917298876394) q[5];
rz(-0.426287700394474) q[5];
ry(1.570667344044619) q[6];
rz(1.0716538685145895) q[6];
ry(-1.5704544382487573) q[7];
rz(-1.477748184397023) q[7];
ry(0.07656551972946879) q[8];
rz(-2.6970616293761314) q[8];
ry(-3.1053803006922287) q[9];
rz(0.0003232253530107698) q[9];
ry(0.000636173688390862) q[10];
rz(2.1776965823874876) q[10];
ry(0.0007611340605429811) q[11];
rz(0.7125255362354821) q[11];
ry(1.5709232738852714) q[12];
rz(-0.20969780341744984) q[12];
ry(-1.570531653872313) q[13];
rz(-1.4221514385291245) q[13];
ry(-1.5665576028412458) q[14];
rz(0.7143264487552853) q[14];
ry(1.5821283723598516) q[15];
rz(-1.3701654120287816) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(-0.10851890978251345) q[0];
rz(1.629780912717691) q[0];
ry(1.5760887637632688) q[1];
rz(0.8189526901581288) q[1];
ry(1.5674694318813347) q[2];
rz(-3.128851778201464) q[2];
ry(1.571461372844476) q[3];
rz(0.006518968848850727) q[3];
ry(3.0806949928391005) q[4];
rz(1.3161034288306943) q[4];
ry(-3.1299639056094493) q[5];
rz(-2.656110740000674) q[5];
ry(-3.1380578082932535) q[6];
rz(1.3360588836747036) q[6];
ry(3.123036129009019) q[7];
rz(-2.7937173832736293) q[7];
ry(0.0001858802265297788) q[8];
rz(1.125385944555636) q[8];
ry(-1.74200935643119) q[9];
rz(1.5711044374634844) q[9];
ry(0.2567156054302337) q[10];
rz(2.8147772837570963) q[10];
ry(0.09734909041943399) q[11];
rz(0.9043603225811818) q[11];
ry(1.5538071346275595) q[12];
rz(0.16881499250959953) q[12];
ry(-1.4887477283113064) q[13];
rz(-2.882817237029148) q[13];
ry(-2.40745376363643) q[14];
rz(-2.7559748545317504) q[14];
ry(2.8981767163497376) q[15];
rz(-2.7327729382651444) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.8260093436335119) q[0];
rz(-1.5366416890924592) q[0];
ry(0.04941581653744327) q[1];
rz(2.309554069046299) q[1];
ry(1.5696467806391692) q[2];
rz(-3.1142465608741254) q[2];
ry(-1.5947695967049338) q[3];
rz(-3.0454726612148284) q[3];
ry(-0.0001651734421970126) q[4];
rz(0.36791785588478915) q[4];
ry(3.141232667388371) q[5];
rz(2.4346664464099295) q[5];
ry(3.141429815019329) q[6];
rz(0.8975844101118866) q[6];
ry(3.137385479148427) q[7];
rz(-1.2546664568245478) q[7];
ry(-1.5708326578671914) q[8];
rz(1.5753990141761056) q[8];
ry(1.5691274392562908) q[9];
rz(0.022494070919444942) q[9];
ry(3.14021498846027) q[10];
rz(-1.3870192302965259) q[10];
ry(3.141297535588382) q[11];
rz(-0.6373947665435749) q[11];
ry(-3.1415857488513566) q[12];
rz(0.2534100862697627) q[12];
ry(-0.00019936145518428816) q[13];
rz(1.5456745219354115) q[13];
ry(0.026799396693504857) q[14];
rz(-1.3903447104194804) q[14];
ry(3.0098376571683225) q[15];
rz(1.9358508278907518) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
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
ry(1.5637596092839852) q[0];
rz(-3.13292231675837) q[0];
ry(1.5697641616099256) q[1];
rz(3.130028298562027) q[1];
ry(1.4871809442940571) q[2];
rz(0.2288835785739307) q[2];
ry(-1.7273125293703178) q[3];
rz(2.0396704834214603) q[3];
ry(-1.5143219980131197) q[4];
rz(-3.0161950762745517) q[4];
ry(-1.5049616335833031) q[5];
rz(1.7611612731602568) q[5];
ry(0.05732093396489435) q[6];
rz(-2.1272050212474287) q[6];
ry(0.22331398434218908) q[7];
rz(-1.627472775584077) q[7];
ry(1.8693720416968898) q[8];
rz(1.5719506247499022) q[8];
ry(1.5703573411774359) q[9];
rz(0.0016036057211819) q[9];
ry(-0.002807898556513422) q[10];
rz(-2.0893461836291207) q[10];
ry(0.0050274979981947965) q[11];
rz(-1.6080842509331186) q[11];
ry(3.1250744792669067) q[12];
rz(-3.1277616929231757) q[12];
ry(-1.6186527185204467) q[13];
rz(0.38096656668843953) q[13];
ry(-1.16770779241355) q[14];
rz(-0.8545893776415711) q[14];
ry(-3.102943295494622) q[15];
rz(-3.0055770885992916) q[15];