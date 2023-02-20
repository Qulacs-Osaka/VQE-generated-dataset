OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.0654014168440433) q[0];
rz(-1.241135200264979) q[0];
ry(-0.0014062352862227456) q[1];
rz(-1.1569078303324236) q[1];
ry(-3.141508784364113) q[2];
rz(0.6446766804575033) q[2];
ry(1.5722399783978203) q[3];
rz(2.5632284990711938) q[3];
ry(-1.7706927354830544) q[4];
rz(1.4764536594255704) q[4];
ry(2.062766463312358) q[5];
rz(-2.8620835108823712) q[5];
ry(0.7637151797775656) q[6];
rz(0.5885800006758352) q[6];
ry(0.35392110676998634) q[7];
rz(-0.3173719238492838) q[7];
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
ry(-0.6180901112497322) q[0];
rz(-2.119299495302803) q[0];
ry(-1.5713818935529682) q[1];
rz(-2.9273253214494184) q[1];
ry(-6.079073707536031e-06) q[2];
rz(1.8924791777344836) q[2];
ry(3.1392715544064695) q[3];
rz(-2.957029154190799) q[3];
ry(1.0551695864163817) q[4];
rz(1.1197123480280382) q[4];
ry(-3.1405011644073233) q[5];
rz(0.1878844465021725) q[5];
ry(-1.6746364341432356) q[6];
rz(0.5167252591791421) q[6];
ry(-2.515212852986078) q[7];
rz(-2.5187428770427083) q[7];
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
ry(-1.570870444506744) q[0];
rz(-0.003957230167979553) q[0];
ry(1.5227808831525078) q[1];
rz(-1.545190818333371) q[1];
ry(-1.5708302264577878) q[2];
rz(1.5408285194409421) q[2];
ry(2.8377074544522602) q[3];
rz(1.1835090536746726) q[3];
ry(-0.6381669327805563) q[4];
rz(2.5824741193305045) q[4];
ry(-2.414946403227319) q[5];
rz(2.713837568749904) q[5];
ry(-3.11901176821526) q[6];
rz(0.19906100886425213) q[6];
ry(1.1546324096309766) q[7];
rz(1.4878590753581618) q[7];
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
ry(1.57079655661232) q[0];
rz(-1.9105817144714818) q[0];
ry(-1.4612853596597184) q[1];
rz(-1.8014351100871282) q[1];
ry(-1.718514158716443) q[2];
rz(2.2258263849737143) q[2];
ry(3.1415633946942094) q[3];
rz(-2.738746441031286) q[3];
ry(-0.00013581737797069136) q[4];
rz(1.377214405407055) q[4];
ry(3.1377774693860196) q[5];
rz(-1.147356019522286) q[5];
ry(2.9285274494143083) q[6];
rz(0.38766471557037036) q[6];
ry(-0.49983025874377046) q[7];
rz(-1.7407967712184884) q[7];
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
ry(-1.5707755329745394) q[0];
rz(-1.3091911762781956) q[0];
ry(-1.5696126166034414) q[1];
rz(1.9105736967203977) q[1];
ry(-1.8452095570388058) q[2];
rz(1.8550142852801885) q[2];
ry(0.7731273107351351) q[3];
rz(-2.5867947795532373) q[3];
ry(-6.603289798613766e-05) q[4];
rz(-0.09783348755710478) q[4];
ry(0.09959806341671153) q[5];
rz(1.4700821652651095) q[5];
ry(3.108109887757234) q[6];
rz(2.136812473915208) q[6];
ry(-0.13422650827320037) q[7];
rz(-2.7624855073501355) q[7];
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
ry(0.003661322397362455) q[0];
rz(-0.26068740837843946) q[0];
ry(1.3335816797662146) q[1];
rz(1.2542821595040436) q[1];
ry(-3.141557835996372) q[2];
rz(-1.5083770785182222) q[2];
ry(-3.141481476728537) q[3];
rz(-2.424094926961936) q[3];
ry(3.1415857734808363) q[4];
rz(1.2938978762087574) q[4];
ry(-3.1409937019230987) q[5];
rz(-2.7734832112805474) q[5];
ry(-1.2956697691858938) q[6];
rz(2.986620010101848) q[6];
ry(1.5547284574349776) q[7];
rz(-0.5038707005772746) q[7];
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
ry(-1.8814571685896049) q[0];
rz(1.470342458789608) q[0];
ry(1.9283159088234392e-05) q[1];
rz(1.5514345170200174) q[1];
ry(-1.5343248702446148) q[2];
rz(-3.0620920352886056) q[2];
ry(1.8096901838126813) q[3];
rz(1.5718536543295465) q[3];
ry(-0.010378951595219732) q[4];
rz(-1.5277305651926962) q[4];
ry(-1.2051294673845918) q[5];
rz(0.3673311402652205) q[5];
ry(0.03243287977888709) q[6];
rz(1.5810843524414309) q[6];
ry(-1.9335443366865661) q[7];
rz(-1.1699519222524897) q[7];
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
ry(-1.5290295301833066) q[0];
rz(3.090008330037864) q[0];
ry(-1.5451712504981936) q[1];
rz(0.5062463576593927) q[1];
ry(-0.0011274718648107624) q[2];
rz(1.0442230189137687) q[2];
ry(1.570523983096339) q[3];
rz(-1.5701166843075358) q[3];
ry(3.141487032012273) q[4];
rz(2.3634647401394644) q[4];
ry(-1.5734250177968816) q[5];
rz(1.4289094560825537) q[5];
ry(-1.6205189780795937) q[6];
rz(0.029603627040090785) q[6];
ry(-2.732495345439844) q[7];
rz(2.602850158084975) q[7];
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
ry(-1.7828376116420381) q[0];
rz(0.5835326758149685) q[0];
ry(-0.6820176341559989) q[1];
rz(-1.5020764551387131) q[1];
ry(1.2380430997207821e-05) q[2];
rz(0.7714247844104171) q[2];
ry(-1.5707947150740207) q[3];
rz(-3.1211764588838147) q[3];
ry(-0.00013755414600879473) q[4];
rz(0.5282509532780618) q[4];
ry(-3.141478203931168) q[5];
rz(-0.14275192689979332) q[5];
ry(1.5144116015362243) q[6];
rz(-0.27127237794234715) q[6];
ry(3.141306349935848) q[7];
rz(1.076619461957214) q[7];
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
ry(-3.141513871936583) q[0];
rz(-0.8014737184107626) q[0];
ry(-2.021726856452668e-05) q[1];
rz(-1.6402782958146878) q[1];
ry(0.00014502649699606602) q[2];
rz(0.9565554485207979) q[2];
ry(-0.001372484442172528) q[3];
rz(1.5504039929798485) q[3];
ry(1.5721285079970992) q[4];
rz(-1.5849669872912395) q[4];
ry(-1.5707970420221897) q[5];
rz(5.857308379053383e-06) q[5];
ry(1.4621331814270742) q[6];
rz(2.340268900843439) q[6];
ry(-0.9030281290704867) q[7];
rz(-2.937711043594147) q[7];
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
ry(-0.6482789488734982) q[0];
rz(-0.06635359816771787) q[0];
ry(-2.459556331511868) q[1];
rz(0.10718732055087887) q[1];
ry(3.1415748865458557) q[2];
rz(-3.0052050573831086) q[2];
ry(1.5708095168846352) q[3];
rz(-0.5568379369348175) q[3];
ry(0.6249920951121357) q[4];
rz(1.5558066565450113) q[4];
ry(-1.5707987809315604) q[5];
rz(-1.6489895885292212) q[5];
ry(0.19998486363778387) q[6];
rz(0.5145600952165211) q[6];
ry(3.1415838888936136) q[7];
rz(0.5608781674825648) q[7];
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
ry(2.870598987236373) q[0];
rz(-2.621626411281223) q[0];
ry(-3.1288759455644106) q[1];
rz(1.0535192760859458) q[1];
ry(-0.16672434287382915) q[2];
rz(-1.6228187652667476) q[2];
ry(-1.5704273084444649) q[3];
rz(1.266881752253216) q[3];
ry(3.1415379533372136) q[4];
rz(-0.4244237387971788) q[4];
ry(-1.5707754832612757) q[5];
rz(1.5703154777213133) q[5];
ry(1.5428383225513969) q[6];
rz(1.51891511351277) q[6];
ry(-3.1415889185130457) q[7];
rz(1.1266936667013818) q[7];
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
ry(-0.4769703290884929) q[0];
rz(1.7474240916896233) q[0];
ry(0.05843635535120582) q[1];
rz(0.9829248199811635) q[1];
ry(3.1415817806621416) q[2];
rz(-1.3348341604288496) q[2];
ry(-4.278695394715283e-06) q[3];
rz(2.7490205511038424) q[3];
ry(-3.1415771677729323) q[4];
rz(-0.6723627757977512) q[4];
ry(-1.5707962042850234) q[5];
rz(-0.3945055318854411) q[5];
ry(1.5665944861677552) q[6];
rz(0.7622469977493316) q[6];
ry(1.610206561629137) q[7];
rz(0.10193235033571037) q[7];
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
ry(2.6785114189970844) q[0];
rz(2.078266535984088) q[0];
ry(1.3276205891028292) q[1];
rz(1.210521412373702) q[1];
ry(1.8447621135952483) q[2];
rz(-0.34129281438487463) q[2];
ry(1.8195999599903068) q[3];
rz(-1.9169525911173277) q[3];
ry(-1.1057697463297398) q[4];
rz(-0.26820003178075674) q[4];
ry(0.26499282209027886) q[5];
rz(-1.6234950589026955) q[5];
ry(0.42214737943164415) q[6];
rz(-2.94391112030765) q[6];
ry(-2.89537846823895) q[7];
rz(-1.8640021843704648) q[7];