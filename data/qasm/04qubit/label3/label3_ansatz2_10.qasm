OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.703336651060374) q[0];
rz(-0.07563820646956289) q[0];
ry(-1.3057605404200472) q[1];
rz(2.1459985125358747) q[1];
ry(-0.21429975459321682) q[2];
rz(-1.8414258665801533) q[2];
ry(-2.755765114779106) q[3];
rz(-0.2692190360434416) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.3671745376582773) q[0];
rz(2.9814012048544054) q[0];
ry(0.06953474599405368) q[1];
rz(-1.123768759198556) q[1];
ry(1.7091514276558666) q[2];
rz(-2.332367274826661) q[2];
ry(-2.942578242431386) q[3];
rz(1.395930148404421) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.2440759337076246) q[0];
rz(-0.10449366692412543) q[0];
ry(-0.06142213758301551) q[1];
rz(-0.09760686828103804) q[1];
ry(-1.6539274012164746) q[2];
rz(-2.3193265616587375) q[2];
ry(-2.0586111458278844) q[3];
rz(-2.330827201512581) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.6531129204895345) q[0];
rz(1.6132843040285814) q[0];
ry(2.243218830155953) q[1];
rz(0.3770245865101232) q[1];
ry(0.6228520013038547) q[2];
rz(1.6653035825084284) q[2];
ry(-2.0758627706846173) q[3];
rz(1.4134182335516883) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.3025362485608571) q[0];
rz(1.3268621700570966) q[0];
ry(-1.0309740352659622) q[1];
rz(1.7800516794113734) q[1];
ry(2.0824335588632836) q[2];
rz(-2.773428663052665) q[2];
ry(-2.7499609778200553) q[3];
rz(2.295927738689837) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.048084745930129354) q[0];
rz(-2.2790574077392995) q[0];
ry(2.09190904320304) q[1];
rz(0.2799572209612701) q[1];
ry(-2.2254442220766224) q[2];
rz(2.3524843636813313) q[2];
ry(2.6930640290599257) q[3];
rz(1.4992749587437677) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.5679475024988233) q[0];
rz(-1.346347829305316) q[0];
ry(0.4732342525754357) q[1];
rz(1.8191713388028234) q[1];
ry(0.28087576161881067) q[2];
rz(2.8022547556052153) q[2];
ry(1.1632392918224286) q[3];
rz(1.62189021774416) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.5705126184032032) q[0];
rz(-0.8249006556258847) q[0];
ry(-2.390339290744958) q[1];
rz(2.718614423296087) q[1];
ry(-1.37837310132159) q[2];
rz(-1.3397609053595163) q[2];
ry(0.2678703586353224) q[3];
rz(1.519460029815386) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.6880457060396632) q[0];
rz(-2.8549351648229586) q[0];
ry(0.9944031160036098) q[1];
rz(1.756333387389277) q[1];
ry(-1.4939649479001584) q[2];
rz(0.45704974914138496) q[2];
ry(-0.503937715420049) q[3];
rz(-2.9651454222493716) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.3598757214425446) q[0];
rz(2.743449675063628) q[0];
ry(-0.17959759882051876) q[1];
rz(-2.9174957043006544) q[1];
ry(2.055704053065941) q[2];
rz(-2.7207203842339402) q[2];
ry(-0.8244823589057821) q[3];
rz(-1.3961521896922762) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.7518440013918464) q[0];
rz(-1.9414347179223976) q[0];
ry(-1.3137455344768387) q[1];
rz(1.1477120007637318) q[1];
ry(0.9484745452184811) q[2];
rz(3.05668513037689) q[2];
ry(1.7702537755177374) q[3];
rz(0.6974592224637917) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.4308428784415335) q[0];
rz(-0.6491654943431315) q[0];
ry(0.7824056091700387) q[1];
rz(0.35315245350991104) q[1];
ry(1.087545879808878) q[2];
rz(-2.7928550277824726) q[2];
ry(-1.706258121719296) q[3];
rz(0.017257796628595508) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.356053325521556) q[0];
rz(0.6971463087650243) q[0];
ry(1.4928316478875567) q[1];
rz(2.5280822243864383) q[1];
ry(2.226439293191674) q[2];
rz(-0.35108958105536137) q[2];
ry(-2.1885804594785085) q[3];
rz(-1.6980977181988959) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.522398736958746) q[0];
rz(-0.8177678733959678) q[0];
ry(3.092529533732027) q[1];
rz(-1.4140603696749228) q[1];
ry(0.3564910190422612) q[2];
rz(-1.3161567275511263) q[2];
ry(1.1497726279644678) q[3];
rz(-1.9765634927613513) q[3];