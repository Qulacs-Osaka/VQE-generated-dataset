OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.564620465171947) q[0];
rz(-0.07168632842690846) q[0];
ry(-2.665984812720668) q[1];
rz(0.4639612387830018) q[1];
ry(1.3874874556926973) q[2];
rz(-2.8063049183592534) q[2];
ry(-1.4404528848526823) q[3];
rz(0.05451089282947207) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.1159769894444995) q[0];
rz(1.5586975549515993) q[0];
ry(-1.48367009750846) q[1];
rz(0.801173949288243) q[1];
ry(2.577961328881583) q[2];
rz(1.3999699825654421) q[2];
ry(-2.80409273122235) q[3];
rz(1.246288777635125) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.8421987207047277) q[0];
rz(-0.9129850411384428) q[0];
ry(-0.39842544842734645) q[1];
rz(1.4342554754961196) q[1];
ry(2.416056217517247) q[2];
rz(2.1891907906619696) q[2];
ry(-2.5690114095799284) q[3];
rz(-1.924790047988659) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.20649374117544) q[0];
rz(1.2977588412339895) q[0];
ry(-1.6883478191769405) q[1];
rz(-2.2758568609578793) q[1];
ry(-1.8259933333223994) q[2];
rz(-1.8867243925310946) q[2];
ry(-2.755588282585753) q[3];
rz(-1.094663754991202) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.36197138109044946) q[0];
rz(1.3029851683557345) q[0];
ry(1.6931341184675786) q[1];
rz(-2.6713072072848143) q[1];
ry(1.089174303276233) q[2];
rz(-1.7082934726261767) q[2];
ry(0.13346481011225045) q[3];
rz(-1.667499434927472) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0557057288558092) q[0];
rz(-0.980893928743961) q[0];
ry(2.5542345654244625) q[1];
rz(-1.9654643385216217) q[1];
ry(-2.7599814076636675) q[2];
rz(-0.10030608477104787) q[2];
ry(1.8241749933091738) q[3];
rz(0.7258698627835775) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.984678931864842) q[0];
rz(-1.6444627419877238) q[0];
ry(1.4769809720913163) q[1];
rz(2.721759897245888) q[1];
ry(-0.17992730578158778) q[2];
rz(2.7703837014773702) q[2];
ry(-0.45983589433533023) q[3];
rz(2.5408944307964876) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.10087072807294195) q[0];
rz(1.3953255639324904) q[0];
ry(2.5213624484878743) q[1];
rz(-0.1229853110449632) q[1];
ry(1.2411176920600333) q[2];
rz(-0.7443727852657405) q[2];
ry(-0.6328562489330833) q[3];
rz(2.9591326836525917) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5210883260805373) q[0];
rz(-0.6533215056862058) q[0];
ry(-0.88201544936868) q[1];
rz(-1.4367473052163797) q[1];
ry(0.5197848953608076) q[2];
rz(-2.21680257047682) q[2];
ry(-2.04776093612343) q[3];
rz(2.4141642073133753) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.880551280146489) q[0];
rz(-2.8578259229387672) q[0];
ry(-2.8452254413771816) q[1];
rz(1.9576096683717226) q[1];
ry(-2.1691701566761172) q[2];
rz(-1.163657356153976) q[2];
ry(-0.6044311406923741) q[3];
rz(2.7518705239181585) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.677277981814183) q[0];
rz(-1.8756365729079925) q[0];
ry(1.573661811927414) q[1];
rz(-0.8443645392241564) q[1];
ry(-0.2782029291770142) q[2];
rz(1.1024811688677278) q[2];
ry(-0.1861306427003404) q[3];
rz(-2.8601084438456943) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2733401129997781) q[0];
rz(-1.5054850035098823) q[0];
ry(0.21564090427560442) q[1];
rz(0.7671535279358359) q[1];
ry(-1.8309291572183997) q[2];
rz(-0.10495369613649164) q[2];
ry(2.6622025751197995) q[3];
rz(-1.0118019533811105) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.1973670452954095) q[0];
rz(-2.4557735300804704) q[0];
ry(-1.1404622770819373) q[1];
rz(-0.9233407241995071) q[1];
ry(-1.8506262408490988) q[2];
rz(0.04409253930847879) q[2];
ry(-0.5475774205180164) q[3];
rz(1.9801214180136695) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5624565125203659) q[0];
rz(-1.6596027740472243) q[0];
ry(-1.171512623996666) q[1];
rz(-1.4609916044377584) q[1];
ry(-2.1561829438052067) q[2];
rz(3.091012533229133) q[2];
ry(0.8795324603534989) q[3];
rz(-1.390614637401993) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.650816538912248) q[0];
rz(1.5783609302795116) q[0];
ry(0.17001223473034124) q[1];
rz(-0.6544267183469906) q[1];
ry(-0.6944844052726147) q[2];
rz(0.8684202063421518) q[2];
ry(-1.505681017823298) q[3];
rz(-0.5294772842645532) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.43794144424324) q[0];
rz(0.8199210087547232) q[0];
ry(1.1453970305745198) q[1];
rz(2.349821620408353) q[1];
ry(-1.5049115961286956) q[2];
rz(-2.945611964529849) q[2];
ry(-2.775176367389852) q[3];
rz(-2.0824543820825476) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.0187602119226504) q[0];
rz(-1.8823132929094673) q[0];
ry(0.5117301724736608) q[1];
rz(-0.0522749573604532) q[1];
ry(-0.29086387870607844) q[2];
rz(-0.4512474726237556) q[2];
ry(-1.3454492207416866) q[3];
rz(1.8500511401274604) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7161353041958374) q[0];
rz(2.814891003646328) q[0];
ry(-2.1326498083649708) q[1];
rz(0.505958133337761) q[1];
ry(-2.2949363338588307) q[2];
rz(1.576463770798289) q[2];
ry(-1.7058055388913909) q[3];
rz(-1.789270382468844) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7895105514556273) q[0];
rz(1.7756974142854773) q[0];
ry(-2.4331161412661886) q[1];
rz(-0.9694846141406527) q[1];
ry(-2.7319067956580896) q[2];
rz(-1.769092174513245) q[2];
ry(0.4081040296197545) q[3];
rz(-2.160625169890954) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7602912767174264) q[0];
rz(-1.3171969642844665) q[0];
ry(1.6963703742457843) q[1];
rz(0.8314155201307561) q[1];
ry(-2.451813567766872) q[2];
rz(2.911024489137654) q[2];
ry(-2.6312616775210746) q[3];
rz(-1.6041820372442217) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.0312910465828122) q[0];
rz(-1.6861055382652517) q[0];
ry(1.6702990754090328) q[1];
rz(-2.581922406532568) q[1];
ry(-2.9303830751345274) q[2];
rz(0.19424830423749245) q[2];
ry(0.9284914720896928) q[3];
rz(-2.3581989050832775) q[3];