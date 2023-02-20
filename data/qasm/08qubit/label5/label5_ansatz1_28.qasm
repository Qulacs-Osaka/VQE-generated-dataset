OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.8943699166712369) q[0];
rz(-0.13934794897004463) q[0];
ry(-3.0109275327197507) q[1];
rz(0.6707579266501226) q[1];
ry(1.9320233761191303) q[2];
rz(-1.335962181493922) q[2];
ry(-2.528695940085699) q[3];
rz(-0.4701179611582695) q[3];
ry(1.2025251838397004) q[4];
rz(-2.8184795767542097) q[4];
ry(1.765822372546141) q[5];
rz(0.8238039616170418) q[5];
ry(-1.175628217659804) q[6];
rz(-2.6458082400388188) q[6];
ry(-2.0853244488650997) q[7];
rz(-0.5217499470768168) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.805821951143169) q[0];
rz(0.16747562175369168) q[0];
ry(1.6666764489411603) q[1];
rz(-1.3711132578057716) q[1];
ry(-1.1142225931261196) q[2];
rz(-2.7717311054675435) q[2];
ry(1.783696575148471) q[3];
rz(-1.945798632428395) q[3];
ry(-2.0035763255723955) q[4];
rz(2.3462349281556967) q[4];
ry(-1.5260879210876828) q[5];
rz(-2.9213388934636333) q[5];
ry(1.2881676011549033) q[6];
rz(2.0481560499711002) q[6];
ry(-1.354742437861824) q[7];
rz(2.29055256035807) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.868972549003519) q[0];
rz(0.5472781497870841) q[0];
ry(1.29820664258151) q[1];
rz(0.985957951129115) q[1];
ry(1.5388629133498801) q[2];
rz(1.9644770706614751) q[2];
ry(0.3244958418630897) q[3];
rz(-2.058932316203473) q[3];
ry(-2.320419943896845) q[4];
rz(-0.09841786302925114) q[4];
ry(1.088135342965907) q[5];
rz(1.2146765989401134) q[5];
ry(-0.2301064839866731) q[6];
rz(-0.9752915316640713) q[6];
ry(1.9443892074553757) q[7];
rz(0.47861006012573265) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.680034427186374) q[0];
rz(-0.7419893879131969) q[0];
ry(-0.9616155652873806) q[1];
rz(1.8947359487113256) q[1];
ry(-0.5060989994290912) q[2];
rz(1.1822347976957188) q[2];
ry(-0.7574650783736301) q[3];
rz(-0.7906649370919734) q[3];
ry(-2.4783192400983576) q[4];
rz(-0.40572171314326333) q[4];
ry(0.7869057198289413) q[5];
rz(1.0128916870087217) q[5];
ry(1.6865619638583205) q[6];
rz(-0.8750271697276878) q[6];
ry(2.0153172203384084) q[7];
rz(2.6586227710511934) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.8031269360437114) q[0];
rz(1.1670282612859904) q[0];
ry(-1.5628492261393616) q[1];
rz(-2.4839838572859043) q[1];
ry(2.318833423757702) q[2];
rz(-0.10526999003235675) q[2];
ry(1.9568348195123448) q[3];
rz(1.7612298199879248) q[3];
ry(1.479816614588504) q[4];
rz(2.683481762893835) q[4];
ry(-0.9082820398283976) q[5];
rz(-0.8030773937865291) q[5];
ry(-0.6615391693853186) q[6];
rz(0.1319677486524924) q[6];
ry(1.8028791382017104) q[7];
rz(-2.430767274222323) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.1262474926135688) q[0];
rz(-0.28677671160511053) q[0];
ry(-1.123269287238809) q[1];
rz(-1.166243877080743) q[1];
ry(1.5363117110297928) q[2];
rz(1.4841744990602486) q[2];
ry(-0.35256761191776026) q[3];
rz(-1.974716548738507) q[3];
ry(-2.601453256671034) q[4];
rz(1.341071489251428) q[4];
ry(-0.06954858810607033) q[5];
rz(-0.7542768633255372) q[5];
ry(-2.599837786607222) q[6];
rz(-1.274993705010663) q[6];
ry(-0.6506383913974897) q[7];
rz(-1.747315222415315) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.3235456613851788) q[0];
rz(-0.8671628102013242) q[0];
ry(1.747593846837676) q[1];
rz(-0.9868664663209241) q[1];
ry(1.5362408634095328) q[2];
rz(-1.204562274050187) q[2];
ry(-0.3955081142417563) q[3];
rz(-3.002768984279949) q[3];
ry(-1.5052535053853928) q[4];
rz(-0.9789861914002819) q[4];
ry(2.3109748667143672) q[5];
rz(1.6114632024988769) q[5];
ry(-1.6132886170526124) q[6];
rz(1.8952162018917351) q[6];
ry(2.3572158731271187) q[7];
rz(2.1147327212514258) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.4257948520453905) q[0];
rz(1.2087514709720848) q[0];
ry(1.6278363549482187) q[1];
rz(0.5935996499760319) q[1];
ry(-2.1647382728262947) q[2];
rz(1.0841145998298538) q[2];
ry(1.7681780669835847) q[3];
rz(1.9018974577863386) q[3];
ry(2.044913495562344) q[4];
rz(1.4223255282674603) q[4];
ry(1.2384885988316852) q[5];
rz(0.5399696506174593) q[5];
ry(2.4537759535663834) q[6];
rz(-0.7212088924043187) q[6];
ry(1.374253054612429) q[7];
rz(-2.244527477470231) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.865613301943271) q[0];
rz(2.4963762944276007) q[0];
ry(2.1513307490903237) q[1];
rz(-0.7248335865370941) q[1];
ry(2.230797129984847) q[2];
rz(-2.721413761379726) q[2];
ry(-0.6304823756811819) q[3];
rz(-2.4209149159488508) q[3];
ry(-0.19018861393605135) q[4];
rz(3.0197718522910137) q[4];
ry(-1.7532136762652453) q[5];
rz(2.087658193155476) q[5];
ry(1.5601865133221162) q[6];
rz(-2.2923219579487797) q[6];
ry(-1.8369558420685792) q[7];
rz(0.13602105867142278) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.5319698745758699) q[0];
rz(2.071309128496096) q[0];
ry(1.9667979648476086) q[1];
rz(2.873662726828071) q[1];
ry(-0.8448092320915502) q[2];
rz(-2.85082496562596) q[2];
ry(-2.8416878875708904) q[3];
rz(-0.05581435969426263) q[3];
ry(1.050180886514212) q[4];
rz(-0.5342691747746571) q[4];
ry(-2.900592764696545) q[5];
rz(-1.163008186528769) q[5];
ry(2.1016369289291315) q[6];
rz(0.12471397219568114) q[6];
ry(0.5315335672034021) q[7];
rz(2.404220881274376) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.2341937146182778) q[0];
rz(-0.18045327352329016) q[0];
ry(2.176113414128417) q[1];
rz(2.58964419939241) q[1];
ry(-0.9069656430063855) q[2];
rz(2.827729686512834) q[2];
ry(-2.110169511199752) q[3];
rz(-2.054808009338074) q[3];
ry(2.430249484743841) q[4];
rz(0.4463907318023519) q[4];
ry(-3.0548156348600726) q[5];
rz(-1.4553601501525684) q[5];
ry(1.3090144478302552) q[6];
rz(1.8508815828018228) q[6];
ry(-1.9119522099382245) q[7];
rz(1.1604857226347094) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.3525397502772645) q[0];
rz(2.7613817477487883) q[0];
ry(2.2854170224284327) q[1];
rz(1.190236537339719) q[1];
ry(-1.5516414243145364) q[2];
rz(2.2683513138575004) q[2];
ry(-2.1329513713608415) q[3];
rz(1.3720170796100433) q[3];
ry(2.8734737663784067) q[4];
rz(-1.7353758819237148) q[4];
ry(-2.480850520384571) q[5];
rz(-1.050406094304564) q[5];
ry(-1.6927160233345644) q[6];
rz(-2.285369364066868) q[6];
ry(1.4187293966101282) q[7];
rz(1.8819967544344154) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.5832872030004523) q[0];
rz(-0.9695275705617079) q[0];
ry(1.7476056044556048) q[1];
rz(-2.936712781070923) q[1];
ry(-2.0532298706045164) q[2];
rz(-1.7704939723432416) q[2];
ry(-2.357743569673862) q[3];
rz(1.007999066570373) q[3];
ry(-3.020792102119095) q[4];
rz(0.7211777224688083) q[4];
ry(1.86862078286369) q[5];
rz(0.24572894727585975) q[5];
ry(-1.116415439106847) q[6];
rz(-2.3793972031184616) q[6];
ry(1.1867300030054269) q[7];
rz(0.4902985067850842) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.4246740025000657) q[0];
rz(-0.533557767872265) q[0];
ry(1.947730204661844) q[1];
rz(-0.6247468678790424) q[1];
ry(-2.0718733924435986) q[2];
rz(2.142246328987719) q[2];
ry(2.753674636101501) q[3];
rz(1.7103027884538502) q[3];
ry(2.375868887632071) q[4];
rz(1.538888770986702) q[4];
ry(-2.7630169543666887) q[5];
rz(-1.2525194403237725) q[5];
ry(1.074822203392248) q[6];
rz(2.4598438188519336) q[6];
ry(0.4937957598303842) q[7];
rz(-1.2004572545936387) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.063425781410958) q[0];
rz(-1.0537878498664093) q[0];
ry(1.3881344466448526) q[1];
rz(1.2337376875774595) q[1];
ry(-3.0142853950097543) q[2];
rz(1.428288557836384) q[2];
ry(2.558474732996989) q[3];
rz(1.396547290344006) q[3];
ry(1.69226366983769) q[4];
rz(1.9242582915444009) q[4];
ry(1.9875524249608203) q[5];
rz(1.602367267080824) q[5];
ry(2.302907514089032) q[6];
rz(2.793071998670533) q[6];
ry(1.353788545895192) q[7];
rz(-0.7324192216008781) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.0382585796333053) q[0];
rz(-1.0406959266719298) q[0];
ry(-2.1409098412266676) q[1];
rz(3.013780332923141) q[1];
ry(1.3812016930034112) q[2];
rz(2.429040108036898) q[2];
ry(-0.9683848368076662) q[3];
rz(3.0957013555406476) q[3];
ry(-1.0749557078922798) q[4];
rz(2.0480749783365653) q[4];
ry(0.40065942365721813) q[5];
rz(-0.6176003601433366) q[5];
ry(1.1104383310121295) q[6];
rz(-1.0621432900148529) q[6];
ry(-2.260742495148656) q[7];
rz(-1.890065203979281) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.5310210621822167) q[0];
rz(1.3336231825575595) q[0];
ry(2.628845273025021) q[1];
rz(0.02811580665068902) q[1];
ry(2.7491888432800415) q[2];
rz(2.685294720377506) q[2];
ry(0.9296691080563715) q[3];
rz(-1.633969733767766) q[3];
ry(1.527487396685227) q[4];
rz(1.7783730528696389) q[4];
ry(-0.22222011465405356) q[5];
rz(2.18308136551346) q[5];
ry(1.8136259166965747) q[6];
rz(-2.477751022134642) q[6];
ry(0.20408601293404074) q[7];
rz(1.3704185103378252) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8588378151830993) q[0];
rz(-2.497374000185125) q[0];
ry(2.5145993577265076) q[1];
rz(1.962424198925783) q[1];
ry(-1.3780041956090514) q[2];
rz(3.10491419356551) q[2];
ry(1.6841816964721943) q[3];
rz(-0.5559135629535872) q[3];
ry(-0.888432912529475) q[4];
rz(-2.7330893804356817) q[4];
ry(-2.384226089470108) q[5];
rz(-2.9520294702569703) q[5];
ry(0.28229634037594026) q[6];
rz(0.7307964280985564) q[6];
ry(0.7258797972999587) q[7];
rz(1.493448733467134) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.579051907866135) q[0];
rz(0.3747616414245396) q[0];
ry(0.9170052515011278) q[1];
rz(1.8435418886144084) q[1];
ry(-2.6130040899310276) q[2];
rz(0.30437217351614615) q[2];
ry(1.0432981120479892) q[3];
rz(-2.589864020524813) q[3];
ry(1.0304599150167046) q[4];
rz(-1.6888139005440381) q[4];
ry(2.6956884132771695) q[5];
rz(-2.0911310546673576) q[5];
ry(-2.136493807736091) q[6];
rz(-0.4244300910195846) q[6];
ry(-2.5809038831553663) q[7];
rz(0.002813118643762991) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.779261771400945) q[0];
rz(2.5147589196464466) q[0];
ry(1.4709508772507853) q[1];
rz(0.9846319956860627) q[1];
ry(-1.8477988986247862) q[2];
rz(-0.25540379841035604) q[2];
ry(1.2783460200427292) q[3];
rz(2.805226689604178) q[3];
ry(0.39591404689578036) q[4];
rz(1.6548768462495094) q[4];
ry(-2.893047317638347) q[5];
rz(3.024413259764412) q[5];
ry(2.307744982176687) q[6];
rz(-1.7443517530585864) q[6];
ry(0.8704029831431915) q[7];
rz(-2.140609430602508) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.225485037549226) q[0];
rz(1.7190710935615536) q[0];
ry(-1.914763947854742) q[1];
rz(-1.0549707045835144) q[1];
ry(0.7147402615127687) q[2];
rz(1.9604825025822743) q[2];
ry(-1.1158077567500246) q[3];
rz(-0.94600114135915) q[3];
ry(2.0098472072226947) q[4];
rz(-0.3964002550368315) q[4];
ry(-1.8090261155464502) q[5];
rz(0.3160880542902626) q[5];
ry(1.6750983616171136) q[6];
rz(-2.954775220024882) q[6];
ry(2.6457811541960625) q[7];
rz(0.8911966124677626) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.151893525975667) q[0];
rz(0.7703115897278741) q[0];
ry(-2.520711782862374) q[1];
rz(-1.8545085474292309) q[1];
ry(3.0090691990448484) q[2];
rz(1.831133337248566) q[2];
ry(1.361952816577829) q[3];
rz(-2.0502414713446075) q[3];
ry(-1.822475209998971) q[4];
rz(-1.1207917424924227) q[4];
ry(-1.2444704641624382) q[5];
rz(-2.246701181559422) q[5];
ry(0.07890121867159738) q[6];
rz(-2.151082727761829) q[6];
ry(1.540794436902405) q[7];
rz(-2.0348136314986798) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-3.0487456279875196) q[0];
rz(0.7392865783004101) q[0];
ry(2.5457248465729925) q[1];
rz(-2.007700488140272) q[1];
ry(1.6464784102008716) q[2];
rz(0.7637491338604339) q[2];
ry(-2.4562497363114786) q[3];
rz(0.0064131384324097596) q[3];
ry(-1.930561088742986) q[4];
rz(-0.577587800235063) q[4];
ry(0.26591396125964906) q[5];
rz(-2.1373218564313365) q[5];
ry(0.5666366408778716) q[6];
rz(0.07363186206006045) q[6];
ry(1.0947321974408175) q[7];
rz(1.6837148387552308) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.662055839869799) q[0];
rz(-1.450646120554124) q[0];
ry(-2.5305463413258327) q[1];
rz(2.6029142586381746) q[1];
ry(2.4637901864506713) q[2];
rz(-2.1282490063281365) q[2];
ry(-1.4103080841936686) q[3];
rz(-2.077257606138671) q[3];
ry(1.356352932542877) q[4];
rz(1.14440785106468) q[4];
ry(1.6062611476770243) q[5];
rz(2.8171946856655454) q[5];
ry(-0.655770991074009) q[6];
rz(1.8224873165623299) q[6];
ry(-0.4577966153032864) q[7];
rz(0.5427292284100688) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.588973861241256) q[0];
rz(-2.8412874853428765) q[0];
ry(-1.3231635477235573) q[1];
rz(-1.3920952503988306) q[1];
ry(-2.9101974024684756) q[2];
rz(2.163344903393779) q[2];
ry(-1.8676105251479025) q[3];
rz(-3.0516704655966898) q[3];
ry(-0.8071879200927308) q[4];
rz(-1.575891541694555) q[4];
ry(-2.369939566388119) q[5];
rz(-2.3085619969396607) q[5];
ry(-2.7870060052452965) q[6];
rz(-1.3331474043656293) q[6];
ry(2.5326883607192934) q[7];
rz(-2.2995132835597536) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.516245616649285) q[0];
rz(2.4176858834450345) q[0];
ry(0.24751464894121847) q[1];
rz(-0.7197174806796901) q[1];
ry(-1.4991349152285727) q[2];
rz(1.9169155763169359) q[2];
ry(-1.7662856267221647) q[3];
rz(2.492271000952898) q[3];
ry(-0.5639763637111713) q[4];
rz(-0.15619276217075753) q[4];
ry(1.6600204906390017) q[5];
rz(1.3730517792537895) q[5];
ry(-1.3780082132783793) q[6];
rz(2.9782985834909983) q[6];
ry(1.9818532734615821) q[7];
rz(-2.7985345231268592) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.451513744914149) q[0];
rz(-1.3919127263505982) q[0];
ry(0.4559099121537956) q[1];
rz(-3.0986271724322165) q[1];
ry(-0.019347749433440406) q[2];
rz(1.5605933273328818) q[2];
ry(2.5939920687376676) q[3];
rz(3.0697900870477706) q[3];
ry(0.6196369529086331) q[4];
rz(-2.401395052207067) q[4];
ry(-2.6377702459276953) q[5];
rz(2.2733477272416875) q[5];
ry(2.213371043098787) q[6];
rz(0.7921346175264398) q[6];
ry(-1.7709536610721142) q[7];
rz(-1.9375132690469858) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.657533836005528) q[0];
rz(2.9310368731456666) q[0];
ry(-1.0074631605316124) q[1];
rz(-2.739404529473493) q[1];
ry(-2.12575682791072) q[2];
rz(-1.9188463588516853) q[2];
ry(2.7561243784383818) q[3];
rz(-3.0987070766895046) q[3];
ry(-2.134657384814066) q[4];
rz(2.5538493052747975) q[4];
ry(-0.7180083169765679) q[5];
rz(2.6421565149375006) q[5];
ry(-1.5578392599280366) q[6];
rz(2.943029596388798) q[6];
ry(-2.2050454588349773) q[7];
rz(-1.1019808561899396) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.274441040199762) q[0];
rz(2.4404245786045236) q[0];
ry(2.81276488185882) q[1];
rz(-1.3639449477256806) q[1];
ry(-1.2804464407621134) q[2];
rz(2.227337941527952) q[2];
ry(0.0592970891556079) q[3];
rz(2.0467957404440096) q[3];
ry(-2.639411298514544) q[4];
rz(2.5124754253950616) q[4];
ry(-2.207685444444913) q[5];
rz(0.335145258279896) q[5];
ry(-1.118379503654223) q[6];
rz(2.4433779278743475) q[6];
ry(-1.1364707222485917) q[7];
rz(1.991684536226424) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.6890596565260543) q[0];
rz(1.3053180420826953) q[0];
ry(1.1782771275931467) q[1];
rz(1.460042760307192) q[1];
ry(2.9598143810149713) q[2];
rz(2.320731275921767) q[2];
ry(-2.5183953909533026) q[3];
rz(2.025947215139782) q[3];
ry(3.042176981625365) q[4];
rz(-0.671290441518103) q[4];
ry(2.369772606212258) q[5];
rz(0.49751861912852746) q[5];
ry(-2.28357151512406) q[6];
rz(0.6710321349039037) q[6];
ry(1.8771792625136159) q[7];
rz(-0.07034172577669118) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.0089952652263223) q[0];
rz(0.6385970961256318) q[0];
ry(-1.4899780031936496) q[1];
rz(0.0008948872256523044) q[1];
ry(-2.4874010549574317) q[2];
rz(-2.5175068982190942) q[2];
ry(2.1090062381630146) q[3];
rz(-3.041801978356841) q[3];
ry(-0.3305008267683105) q[4];
rz(-2.209175780068067) q[4];
ry(2.6169753841851717) q[5];
rz(0.5213289111303564) q[5];
ry(-1.6696460980229273) q[6];
rz(0.3368937865332988) q[6];
ry(0.8066745612134056) q[7];
rz(1.900222745213438) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(3.0009821893258004) q[0];
rz(2.487951877546147) q[0];
ry(1.309021734032792) q[1];
rz(0.9271713823462624) q[1];
ry(2.5749917570646566) q[2];
rz(0.4591977567020775) q[2];
ry(-1.3312270318731159) q[3];
rz(-0.2120924638250825) q[3];
ry(-2.7059019739536345) q[4];
rz(2.0106668505407743) q[4];
ry(2.6170794211444868) q[5];
rz(-2.319675911531203) q[5];
ry(2.212496873298438) q[6];
rz(0.14536462427467156) q[6];
ry(-1.3858552968135898) q[7];
rz(1.3143069394072195) q[7];