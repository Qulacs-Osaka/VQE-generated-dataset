OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5632076124934469) q[0];
rz(3.0167374405393863) q[0];
ry(-1.5807292074677626) q[1];
rz(-2.7720818610693643) q[1];
ry(0.0023930767823685506) q[2];
rz(0.012641400718654161) q[2];
ry(1.5708541910321243) q[3];
rz(1.6102775966281948) q[3];
ry(-0.00028473191482550667) q[4];
rz(0.6913428855663063) q[4];
ry(1.5745513031120035) q[5];
rz(0.08520250152201184) q[5];
ry(-0.013086596969007458) q[6];
rz(3.014936303667968) q[6];
ry(-3.1404462638033) q[7];
rz(1.261754578246604) q[7];
ry(2.4512949874668384) q[8];
rz(2.002034451013226) q[8];
ry(3.0385817524841627) q[9];
rz(-2.109940755871582) q[9];
ry(0.8219378291482501) q[10];
rz(-0.3862664831114028) q[10];
ry(-3.1401121197691872) q[11];
rz(-2.3838845381886147) q[11];
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
ry(-0.2705693228272153) q[0];
rz(-2.711014440227379) q[0];
ry(-1.7560977050771962) q[1];
rz(1.9904404382797403) q[1];
ry(-0.006097342877767531) q[2];
rz(1.0692458306652937) q[2];
ry(-1.5697784630611231) q[3];
rz(-1.5964428190202193) q[3];
ry(3.141539780508686) q[4];
rz(0.15278721357148053) q[4];
ry(0.3817688387595606) q[5];
rz(0.5404279341365822) q[5];
ry(1.4223288766419735) q[6];
rz(0.14841793867888917) q[6];
ry(1.5705092083088577) q[7];
rz(2.1747833853558713) q[7];
ry(-2.539777123759471) q[8];
rz(-0.01677068462558019) q[8];
ry(0.04915287163797011) q[9];
rz(-0.39136883199755684) q[9];
ry(-2.146014291662925) q[10];
rz(2.2592320264020955) q[10];
ry(-1.5728477904614522) q[11];
rz(-0.540609596006556) q[11];
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
ry(0.0821234581923802) q[0];
rz(1.260107979241111) q[0];
ry(1.9315246962176764) q[1];
rz(-3.1012891598907713) q[1];
ry(0.020300518453018945) q[2];
rz(0.14585939480814855) q[2];
ry(2.1115041307671634) q[3];
rz(-0.028738727923471933) q[3];
ry(3.1415479355187794) q[4];
rz(-1.903258738769994) q[4];
ry(-3.141575716476138) q[5];
rz(3.0717015417775935) q[5];
ry(3.1198732200594548) q[6];
rz(-1.4705082184107832) q[6];
ry(-3.1009806152386905) q[7];
rz(-2.53716699715292) q[7];
ry(1.5773759039591488) q[8];
rz(-3.0954030924900606) q[8];
ry(1.574362527099502) q[9];
rz(-2.627653091400667) q[9];
ry(-0.002690640972225822) q[10];
rz(-1.6319199107306996) q[10];
ry(-3.13921670230133) q[11];
rz(2.6012535555059615) q[11];
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
ry(1.598682780568259) q[0];
rz(1.6034829113412363) q[0];
ry(1.7656779674777106) q[1];
rz(-0.6712830663292833) q[1];
ry(0.21643627935027787) q[2];
rz(3.132105800771296) q[2];
ry(-1.8981730157893049) q[3];
rz(-1.508992935387973) q[3];
ry(4.4173006259029535e-05) q[4];
rz(-2.1019915833741014) q[4];
ry(-0.00725959201277071) q[5];
rz(0.6140440433111465) q[5];
ry(-0.02213036552596837) q[6];
rz(-1.5221004836382683) q[6];
ry(-0.8786388408021533) q[7];
rz(-0.7532467540342331) q[7];
ry(0.9452685896550318) q[8];
rz(-0.11547580079398401) q[8];
ry(0.83557099783207) q[9];
rz(-0.6973455766616988) q[9];
ry(-1.56719307879335) q[10];
rz(1.183227117591184) q[10];
ry(1.5709102434840414) q[11];
rz(1.058053684312573) q[11];
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
ry(1.6765146334097256) q[0];
rz(-0.08466775014822756) q[0];
ry(3.113920290068586) q[1];
rz(2.438108335796103) q[1];
ry(-1.5714862441467075) q[2];
rz(3.1343956854678092) q[2];
ry(-1.5266546824150442) q[3];
rz(0.5372591666710506) q[3];
ry(0.00020714231820129214) q[4];
rz(-0.051493084672332434) q[4];
ry(3.141589400791813) q[5];
rz(-2.3107411854850985) q[5];
ry(-1.5939563289804684) q[6];
rz(2.950820299979559) q[6];
ry(-3.1074400820242682) q[7];
rz(2.3881899996832323) q[7];
ry(1.5701421011410606) q[8];
rz(3.1404516189977256) q[8];
ry(-1.5735294921013137) q[9];
rz(-2.5109846549448456) q[9];
ry(3.1341341071681534) q[10];
rz(1.0905546538483657) q[10];
ry(0.0016929559430005625) q[11];
rz(3.0196854472919985) q[11];
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
ry(1.150421598760949) q[0];
rz(-2.3185304221740033) q[0];
ry(-2.719397624201502) q[1];
rz(-3.121439302813173) q[1];
ry(1.5744295552548813) q[2];
rz(2.8461880816140326) q[2];
ry(-0.032527012819551615) q[3];
rz(2.52260736485459) q[3];
ry(-1.5708982968730467) q[4];
rz(0.03812241896850387) q[4];
ry(-7.633597045586527e-05) q[5];
rz(0.6678698860705061) q[5];
ry(0.17177757464746388) q[6];
rz(-0.3619939522463662) q[6];
ry(-1.6012897178179675) q[7];
rz(1.0937283500667117) q[7];
ry(-2.285789847686331) q[8];
rz(-1.573135118067321) q[8];
ry(-3.1170260721188043) q[9];
rz(-2.511967683346362) q[9];
ry(-2.7476070667072436) q[10];
rz(-3.0980004884123202) q[10];
ry(-1.5641859474275899) q[11];
rz(-1.5675052785411563) q[11];
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
ry(3.1264934281267767) q[0];
rz(2.348061399123923) q[0];
ry(3.04348229872363) q[1];
rz(-3.1194477126706532) q[1];
ry(1.5691565437692585) q[2];
rz(1.5714465040739904) q[2];
ry(-1.5706897546730374) q[3];
rz(3.1415586441478665) q[3];
ry(-3.1415808689334765) q[4];
rz(1.60885144902516) q[4];
ry(3.1415484155801443) q[5];
rz(-2.280947112802896) q[5];
ry(-6.350375311381384e-05) q[6];
rz(2.129634047576447) q[6];
ry(0.00016587170374076672) q[7];
rz(0.739815310544815) q[7];
ry(1.5668447345108856) q[8];
rz(-1.5700227075835231) q[8];
ry(-1.5707613337282618) q[9];
rz(-1.6181108549865437) q[9];
ry(1.5705791651078163) q[10];
rz(-2.7780318191687092) q[10];
ry(1.5673307067584386) q[11];
rz(-2.6039733315144917) q[11];
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
ry(3.1373652903964495) q[0];
rz(0.464958981410283) q[0];
ry(-1.5081520272594846) q[1];
rz(3.0174578885667827) q[1];
ry(2.9418774663916265) q[2];
rz(0.0006155221899453166) q[2];
ry(-1.5706878725986613) q[3];
rz(-1.7653079144318145) q[3];
ry(1.5685071038215357) q[4];
rz(7.248769301071434e-05) q[4];
ry(-0.00010800206684263932) q[5];
rz(0.2748521007574878) q[5];
ry(0.06736808552508133) q[6];
rz(3.1351164894473738) q[6];
ry(-0.2366718644724557) q[7];
rz(1.288614946958239) q[7];
ry(-1.7094888827611134) q[8];
rz(-1.5671763804597303) q[8];
ry(-1.5616448101562235) q[9];
rz(-1.5687243428138515) q[9];
ry(3.140596074701451) q[10];
rz(-2.775837863649817) q[10];
ry(-0.005949301220392118) q[11];
rz(1.0315554043928852) q[11];
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
ry(-3.139927336965099) q[0];
rz(-1.2547695466994744) q[0];
ry(0.0021473139450204444) q[1];
rz(-1.423189243112691) q[1];
ry(1.5708216610254304) q[2];
rz(1.4937381229101838) q[2];
ry(1.5707605394931392) q[3];
rz(0.9852349414689394) q[3];
ry(1.5707459940554989) q[4];
rz(-2.8384943353996746) q[4];
ry(-0.002685140897688676) q[5];
rz(-2.902795974594355) q[5];
ry(-1.5709629489781545) q[6];
rz(-0.06583779226118837) q[6];
ry(1.570877598042257) q[7];
rz(-3.141506702947155) q[7];
ry(1.5711312072407984) q[8];
rz(1.539842500026781) q[8];
ry(-1.5675178947378092) q[9];
rz(1.569516159837781) q[9];
ry(1.5726231972178266) q[10];
rz(1.5738631547960802) q[10];
ry(2.387869006885369) q[11];
rz(0.5253797161420931) q[11];
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
ry(1.5932528245857984) q[0];
rz(-1.5682723744988687) q[0];
ry(-1.5627666133236866) q[1];
rz(1.476347332506241) q[1];
ry(3.141358517999631) q[2];
rz(-1.582135444948322) q[2];
ry(3.1408190251330006) q[3];
rz(0.056087255791297246) q[3];
ry(-0.00039886361209884313) q[4];
rz(2.838518617802549) q[4];
ry(-1.5712223196801318) q[5];
rz(0.2511906199557385) q[5];
ry(3.141430870425825) q[6];
rz(3.0758225577378444) q[6];
ry(-2.8754092100119237) q[7];
rz(2.9743951895255805e-05) q[7];
ry(-1.570662812486708) q[8];
rz(-1.5530305832996223) q[8];
ry(-1.570305418764506) q[9];
rz(1.5728603957031355) q[9];
ry(-1.570938146810395) q[10];
rz(1.5731461022398925) q[10];
ry(0.00968716857279854) q[11];
rz(-0.5261134193051394) q[11];
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
ry(1.5734358765008811) q[0];
rz(2.2486156691945514) q[0];
ry(3.139887986948213) q[1];
rz(3.0626284641035837) q[1];
ry(1.5705869143382591) q[2];
rz(-1.5707146575522213) q[2];
ry(1.7688494455939145e-05) q[3];
rz(0.9291848421493192) q[3];
ry(-1.5706514684403214) q[4];
rz(-2.976930709087799) q[4];
ry(-3.141465112894361) q[5];
rz(0.2511712806820864) q[5];
ry(1.570994752620754) q[6];
rz(-1.7911937881633584) q[6];
ry(1.570752152287966) q[7];
rz(1.5707315883278616) q[7];
ry(1.5727276663429037) q[8];
rz(1.5734505450797593) q[8];
ry(1.247507999913818) q[9];
rz(1.5645066263687186) q[9];
ry(1.4739943253158674) q[10];
rz(-2.9888451249201022) q[10];
ry(1.5559483427456904) q[11];
rz(-3.138153372353456) q[11];
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
ry(3.1412951671321467) q[0];
rz(0.6777810032955262) q[0];
ry(1.5708561295223564) q[1];
rz(0.043487596652538756) q[1];
ry(1.5707994921185806) q[2];
rz(-3.1414290493980026) q[2];
ry(-1.5716239399014171) q[3];
rz(-3.1403722760400115) q[3];
ry(0.000447576831328841) q[4];
rz(-0.16459719668271952) q[4];
ry(-1.5708135766729203) q[5];
rz(0.002357103414521333) q[5];
ry(3.1414550037716644) q[6];
rz(-1.5150442052799578) q[6];
ry(1.5706475066445293) q[7];
rz(-0.7123505002645123) q[7];
ry(0.9409295416670779) q[8];
rz(1.5670673484802131) q[8];
ry(-1.577662888971949) q[9];
rz(-2.781236001344462) q[9];
ry(-0.0009702103896520597) q[10];
rz(-1.8313004699844173) q[10];
ry(-1.5714017928039778) q[11];
rz(-1.5575686329366416) q[11];
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
ry(1.5706972349612316) q[0];
rz(2.816822716406739) q[0];
ry(-0.009733596757174112) q[1];
rz(1.7712403765792768) q[1];
ry(1.572367598561259) q[2];
rz(-0.7219162852287444) q[2];
ry(1.5707720246656807) q[3];
rz(8.372157080227538e-05) q[3];
ry(-1.5709504068536815) q[4];
rz(3.0702953256375873) q[4];
ry(1.5708140840795985) q[5];
rz(3.1415884541959462) q[5];
ry(3.1411331981051624) q[6];
rz(0.05814601574788635) q[6];
ry(-0.00043570658227043424) q[7];
rz(-2.431080076220176) q[7];
ry(-1.5710065812619172) q[8];
rz(0.08531244025914336) q[8];
ry(9.455216958720314e-06) q[9];
rz(2.7796728087035825) q[9];
ry(0.07966623668111607) q[10];
rz(1.6496843371736227) q[10];
ry(1.5654683982979716) q[11];
rz(1.5716993097343614) q[11];
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
ry(-0.0013911195413722766) q[0];
rz(2.0441698884765773) q[0];
ry(0.00038056776970751827) q[1];
rz(0.7962486006663875) q[1];
ry(-3.141295333330545) q[2];
rz(0.9980582305707637) q[2];
ry(1.571802857944638) q[3];
rz(2.6110161261045137) q[3];
ry(3.1415906043463178) q[4];
rz(-3.0637862932601223) q[4];
ry(-1.5710271293352038) q[5];
rz(1.0404475515740643) q[5];
ry(0.0001237639716711314) q[6];
rz(0.3671524809380289) q[6];
ry(-1.5710185337195384) q[7];
rz(2.611265580852338) q[7];
ry(-3.141431984543878) q[8];
rz(-1.3364184535946464) q[8];
ry(-1.5707663420824733) q[9];
rz(-2.1010090133413653) q[9];
ry(1.5707109283198895) q[10];
rz(-1.4217464210615798) q[10];
ry(1.570936695099328) q[11];
rz(2.6113200800096155) q[11];