OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.2237267845525057) q[0];
rz(-1.0032688266262035) q[0];
ry(-0.6940103938108738) q[1];
rz(-1.6253333996187367) q[1];
ry(-0.10430239423673537) q[2];
rz(-3.1338814985107044) q[2];
ry(-2.414262143845154) q[3];
rz(0.5678375710052347) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5423686537981066) q[0];
rz(1.462981790170925) q[0];
ry(-1.2679878179765978) q[1];
rz(2.7895206267155612) q[1];
ry(-0.5281942899926207) q[2];
rz(2.3916156536818365) q[2];
ry(-2.1197774236107056) q[3];
rz(0.3038792873397711) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.29739733425663406) q[0];
rz(2.680947788973466) q[0];
ry(-3.0001763617753534) q[1];
rz(-2.6598340990715967) q[1];
ry(2.3514184681369295) q[2];
rz(1.276665442209187) q[2];
ry(-2.115575925334433) q[3];
rz(-1.1527408474311462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.05092429215753712) q[0];
rz(1.2287073382175389) q[0];
ry(2.9540775798848005) q[1];
rz(1.731647331705405) q[1];
ry(-0.7888567123281413) q[2];
rz(2.676998014614632) q[2];
ry(-1.2866262960626065) q[3];
rz(1.9073165037535065) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.053521834875613) q[0];
rz(0.3665785247645461) q[0];
ry(-0.11447445314381352) q[1];
rz(2.5385304100741966) q[1];
ry(-0.524692632361071) q[2];
rz(1.0482517766668336) q[2];
ry(2.259995739237588) q[3];
rz(0.40747874866821127) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.589981026156387) q[0];
rz(-0.5565606807735667) q[0];
ry(2.704858836275029) q[1];
rz(2.3585299366139467) q[1];
ry(0.32807819019521034) q[2];
rz(-3.12719222335652) q[2];
ry(1.8589975507566883) q[3];
rz(-1.7905594295459177) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.6814171624186955) q[0];
rz(-2.8632201745900927) q[0];
ry(0.8169000004301452) q[1];
rz(-3.0836978116824145) q[1];
ry(0.33722470831526297) q[2];
rz(2.531965264599212) q[2];
ry(0.6573542223776043) q[3];
rz(0.8384378128376067) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.090832609132184) q[0];
rz(-2.0566049282288397) q[0];
ry(-0.7109487112384335) q[1];
rz(-0.9139140422503038) q[1];
ry(-2.6865046351947) q[2];
rz(-1.100893960310075) q[2];
ry(2.115861721781765) q[3];
rz(0.4722835838222086) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.844164090225493) q[0];
rz(-2.136093931393665) q[0];
ry(2.252277885344042) q[1];
rz(-1.0864075218301261) q[1];
ry(-0.48390021897828456) q[2];
rz(0.521961573522191) q[2];
ry(-1.6561209959816952) q[3];
rz(-0.8326667729626643) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.060787460975340046) q[0];
rz(1.3248818770685888) q[0];
ry(-1.5632068842977036) q[1];
rz(2.8154887727461273) q[1];
ry(1.3362932465587365) q[2];
rz(1.3060163282721045) q[2];
ry(-2.548750539409626) q[3];
rz(1.9322237688011092) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.2796282388846887) q[0];
rz(0.170107832509476) q[0];
ry(-2.342609196179136) q[1];
rz(-0.9727770508809722) q[1];
ry(0.8579764735707034) q[2];
rz(2.1570314458058126) q[2];
ry(2.4751542289277118) q[3];
rz(1.184459405676545) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5015668271645082) q[0];
rz(-1.0382399078115556) q[0];
ry(0.7716581149635939) q[1];
rz(-2.278158308286476) q[1];
ry(-1.9626115351127318) q[2];
rz(0.5805381158360974) q[2];
ry(-1.3223528410301464) q[3];
rz(1.939791453547735) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.80233473355056) q[0];
rz(-1.8592264482006728) q[0];
ry(-1.5388795722706767) q[1];
rz(0.7567626616879572) q[1];
ry(-2.241373488952364) q[2];
rz(-1.3157016511669208) q[2];
ry(-0.4719800037435647) q[3];
rz(-1.9607924409650144) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7919744213911697) q[0];
rz(1.7418042765708268) q[0];
ry(0.2889500563996359) q[1];
rz(-1.2028040152263584) q[1];
ry(2.9396256411818236) q[2];
rz(-1.3154224102402197) q[2];
ry(2.5678492439421414) q[3];
rz(1.8730905081956177) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8647854952094542) q[0];
rz(3.001158819263974) q[0];
ry(0.21944041781972867) q[1];
rz(-1.5407782170034583) q[1];
ry(1.0523402189737077) q[2];
rz(2.463157206361961) q[2];
ry(2.3209572290972407) q[3];
rz(1.9360090184389236) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.14532990820614458) q[0];
rz(1.0094889880337856) q[0];
ry(-1.0267447921097412) q[1];
rz(-0.8533935972232065) q[1];
ry(-0.9980314289205838) q[2];
rz(-2.2128228273324417) q[2];
ry(-0.4963454556200416) q[3];
rz(-1.1008020820974358) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6377025086984197) q[0];
rz(1.38665544387813) q[0];
ry(0.7617239955464532) q[1];
rz(-1.8858729129829834) q[1];
ry(2.4038159250362376) q[2];
rz(-1.5397790774298423) q[2];
ry(0.5458607313256829) q[3];
rz(-1.706149516516862) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7024445030999953) q[0];
rz(-2.1464632200152387) q[0];
ry(1.156821296060567) q[1];
rz(2.7502205638693864) q[1];
ry(-1.1394864190586267) q[2];
rz(-2.469356522415323) q[2];
ry(0.1213936796514348) q[3];
rz(-2.624134187278785) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.44928308121824934) q[0];
rz(2.489132430284816) q[0];
ry(-0.9214044797687776) q[1];
rz(0.8436501270809816) q[1];
ry(2.750135816498783) q[2];
rz(-1.8553925001871991) q[2];
ry(-0.45175795817186337) q[3];
rz(0.12046866226650532) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1398190533287007) q[0];
rz(-0.666053429523231) q[0];
ry(1.8095174262262295) q[1];
rz(2.945139834956561) q[1];
ry(-2.415991461025216) q[2];
rz(-1.3323490962700806) q[2];
ry(-2.1283360019605775) q[3];
rz(2.5658605845021274) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.31448822618199124) q[0];
rz(-2.3223961072047916) q[0];
ry(0.8718084380965623) q[1];
rz(-3.0945379784145794) q[1];
ry(0.3360258135042511) q[2];
rz(2.3274420686680126) q[2];
ry(2.7156694748636854) q[3];
rz(1.3638839628042205) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1986974073557184) q[0];
rz(-0.7223919297291674) q[0];
ry(-0.6775849327208965) q[1];
rz(-0.7323014289486568) q[1];
ry(-0.2550273247572994) q[2];
rz(0.318476875034408) q[2];
ry(0.2715598126330745) q[3];
rz(0.22476894204444606) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.8152830962128434) q[0];
rz(-1.0343051354532795) q[0];
ry(2.748548553189556) q[1];
rz(-0.5672719291223095) q[1];
ry(-1.7102431989456581) q[2];
rz(2.6817397902367475) q[2];
ry(0.8854634933123329) q[3];
rz(1.6123047569083562) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.663371128885654) q[0];
rz(-2.4174735256628512) q[0];
ry(-2.571450580192964) q[1];
rz(0.08220689337286567) q[1];
ry(-1.4899216200488745) q[2];
rz(1.3163192958069123) q[2];
ry(2.103511307978385) q[3];
rz(-0.06650394462316861) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.606194663833787) q[0];
rz(1.8563827447951242) q[0];
ry(-2.2424559916801767) q[1];
rz(3.0235608572883446) q[1];
ry(-0.700309500175847) q[2];
rz(-0.24064201327870993) q[2];
ry(3.0673989449679206) q[3];
rz(-2.2520619211287256) q[3];