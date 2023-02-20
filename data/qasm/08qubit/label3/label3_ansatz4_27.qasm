OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(3.0013515952073986) q[0];
rz(2.6106262464698244) q[0];
ry(2.312603431086303) q[1];
rz(-2.907753095483151) q[1];
ry(2.6416133323463953) q[2];
rz(-1.3957704929651993) q[2];
ry(1.1468893394880364) q[3];
rz(2.943309476324173) q[3];
ry(2.0885888465878963) q[4];
rz(3.087928403907778) q[4];
ry(0.8348571015879949) q[5];
rz(1.8225653144336003) q[5];
ry(2.336364807753481) q[6];
rz(-2.4529397749158024) q[6];
ry(1.831419363601295) q[7];
rz(-1.6268354011583044) q[7];
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
ry(1.6274876012403388) q[0];
rz(-0.19467511541463087) q[0];
ry(2.8882482627731854) q[1];
rz(0.25960403952261346) q[1];
ry(1.4924718667074997) q[2];
rz(3.0590673924014222) q[2];
ry(-1.386165015043835) q[3];
rz(-0.1624429030985599) q[3];
ry(-1.8406699998455913) q[4];
rz(-0.6788292625613348) q[4];
ry(-1.3347921496146897) q[5];
rz(1.7314649018698427) q[5];
ry(2.8416408039420773) q[6];
rz(-0.8118805999088261) q[6];
ry(2.4479424030156625) q[7];
rz(1.8102830446066376) q[7];
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
ry(2.638189442401965) q[0];
rz(0.09832227257492375) q[0];
ry(0.4968461543162208) q[1];
rz(1.953014345099687) q[1];
ry(-1.5781187272325043) q[2];
rz(-0.030135223672447253) q[2];
ry(3.0877358129079018) q[3];
rz(0.003952238857429791) q[3];
ry(-1.9263565547332986) q[4];
rz(-2.9141282987489494) q[4];
ry(-0.30960130401426955) q[5];
rz(1.9780550373909707) q[5];
ry(-1.5885883351424441) q[6];
rz(0.06090505847830663) q[6];
ry(-2.148210765279498) q[7];
rz(-2.6166061570153594) q[7];
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
ry(-1.1173875577598336) q[0];
rz(2.032124925143668) q[0];
ry(-0.8403364564590522) q[1];
rz(0.1343437902406759) q[1];
ry(-1.9291634192286136) q[2];
rz(-1.5756708102172121) q[2];
ry(-1.1723339048369015) q[3];
rz(0.685813257444842) q[3];
ry(2.1508397318498194) q[4];
rz(1.6820829567323061) q[4];
ry(-0.1306562191052033) q[5];
rz(-2.584631306419837) q[5];
ry(0.6220738976995612) q[6];
rz(1.0041094681978748) q[6];
ry(-0.6334181535417941) q[7];
rz(0.8959941602197787) q[7];
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
ry(-2.283351005600981) q[0];
rz(-0.2912474287824934) q[0];
ry(1.5848786078406458) q[1];
rz(-0.17391157232756263) q[1];
ry(0.2852358879933467) q[2];
rz(-2.2448702346824865) q[2];
ry(-2.5396414042482665) q[3];
rz(1.949120462139614) q[3];
ry(-2.4317943569397125) q[4];
rz(-0.40272678414479) q[4];
ry(2.2194372127684883) q[5];
rz(-1.7337922680982922) q[5];
ry(-2.689402003398262) q[6];
rz(-2.107487409376019) q[6];
ry(0.48273581101880614) q[7];
rz(-1.4956237844035953) q[7];
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
ry(2.682948352068345) q[0];
rz(0.9914416061733284) q[0];
ry(-0.10278071800423481) q[1];
rz(-1.0701405008143539) q[1];
ry(-2.155894627804181) q[2];
rz(-2.8865000037335338) q[2];
ry(-2.0987021785959703) q[3];
rz(-0.7346655242711557) q[3];
ry(0.904252315429054) q[4];
rz(-2.094363449410677) q[4];
ry(2.0167343686028705) q[5];
rz(2.659749651676946) q[5];
ry(-2.524216182905221) q[6];
rz(-1.1231996747142698) q[6];
ry(-0.9399162324338884) q[7];
rz(-2.149290284929621) q[7];
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
ry(2.06276299144451) q[0];
rz(-0.849931549555017) q[0];
ry(-0.9417084308310084) q[1];
rz(0.7782869332063189) q[1];
ry(-2.3004089945625825) q[2];
rz(1.3031627231271277) q[2];
ry(-0.2768759598911016) q[3];
rz(2.9456776402304303) q[3];
ry(0.3969161473325487) q[4];
rz(-1.4749564642755688) q[4];
ry(-1.4346986749915986) q[5];
rz(-0.6616589466922389) q[5];
ry(-2.853337783127127) q[6];
rz(0.13145248146776553) q[6];
ry(2.4112268827513175) q[7];
rz(0.5252680212534275) q[7];
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
ry(-1.1059698531559434) q[0];
rz(-1.6439571577689447) q[0];
ry(-3.0837712099564683) q[1];
rz(1.2105627141070643) q[1];
ry(2.901010026633398) q[2];
rz(-2.328907788193858) q[2];
ry(0.36969050241388035) q[3];
rz(-2.4758237229391455) q[3];
ry(1.5034786435870668) q[4];
rz(0.09715619032303913) q[4];
ry(2.5332966288221472) q[5];
rz(-0.07591715067929794) q[5];
ry(-2.8727123131612324) q[6];
rz(-1.9300646127599685) q[6];
ry(1.964513832707204) q[7];
rz(-0.1553103430073559) q[7];
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
ry(2.757265227672943) q[0];
rz(2.858052828164296) q[0];
ry(2.7233383770412063) q[1];
rz(-2.881367048041809) q[1];
ry(0.2945482534337911) q[2];
rz(-0.8023107693791802) q[2];
ry(2.203274982336969) q[3];
rz(-0.869895150970003) q[3];
ry(-3.0345105998672017) q[4];
rz(-1.6915451241481119) q[4];
ry(-0.7107824376727873) q[5];
rz(3.101698750410165) q[5];
ry(0.3202279047618166) q[6];
rz(0.7922110171960346) q[6];
ry(1.1315764035250528) q[7];
rz(-1.1254563355674048) q[7];
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
ry(2.709835594837287) q[0];
rz(1.6591974967821335) q[0];
ry(0.5128244360285024) q[1];
rz(-1.7885281893317986) q[1];
ry(-3.1034858121941276) q[2];
rz(1.6898302331809143) q[2];
ry(-0.49886808063619914) q[3];
rz(-2.7803653589462276) q[3];
ry(-2.4029139073556567) q[4];
rz(-0.34647347765633235) q[4];
ry(0.5011524542087302) q[5];
rz(-1.64140334166811) q[5];
ry(1.103057873764671) q[6];
rz(0.2853270146526143) q[6];
ry(1.1041391272474694) q[7];
rz(-0.384337744052388) q[7];
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
ry(-1.2473410831405742) q[0];
rz(1.7909342890354578) q[0];
ry(1.1119305394711656) q[1];
rz(-1.9941260917886514) q[1];
ry(1.316082247103493) q[2];
rz(2.9912102571761587) q[2];
ry(1.2878111852037382) q[3];
rz(3.0632956034634296) q[3];
ry(0.5655161169545495) q[4];
rz(0.3991191813767463) q[4];
ry(0.056188586692495456) q[5];
rz(1.4144680790624076) q[5];
ry(0.4666275884237638) q[6];
rz(-0.8221834391639703) q[6];
ry(-2.4211133683256216) q[7];
rz(1.0599114957793327) q[7];
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
ry(-2.717813240210792) q[0];
rz(0.13600859862494558) q[0];
ry(-2.61859402739537) q[1];
rz(-3.072456924948387) q[1];
ry(1.6032035809489695) q[2];
rz(-1.250702680297067) q[2];
ry(-3.024547816800123) q[3];
rz(-1.92125053331414) q[3];
ry(-2.2422788718569544) q[4];
rz(0.12168684799522952) q[4];
ry(1.447624094271242) q[5];
rz(-0.6969472789877643) q[5];
ry(-1.5673686699940046) q[6];
rz(0.02092299013483002) q[6];
ry(0.10178854559783801) q[7];
rz(1.9098637692700224) q[7];
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
ry(1.3013601024216463) q[0];
rz(-2.7807854229340934) q[0];
ry(-0.17535079519138908) q[1];
rz(-1.357693182976651) q[1];
ry(2.2002870606375273) q[2];
rz(1.466694712389904) q[2];
ry(0.9735275641088252) q[3];
rz(-0.2735728866371491) q[3];
ry(-2.8387058291765896) q[4];
rz(-0.34424166321201793) q[4];
ry(-1.8695568425214173) q[5];
rz(-1.6929337464887995) q[5];
ry(0.3199666272266679) q[6];
rz(2.5307899060360604) q[6];
ry(-2.8286290544127946) q[7];
rz(2.163797480263903) q[7];
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
ry(1.2696309713641514) q[0];
rz(2.9197635162727678) q[0];
ry(1.2145616239062855) q[1];
rz(1.7763387127560621) q[1];
ry(0.7717080963800287) q[2];
rz(-2.2659054382790993) q[2];
ry(-2.896845886083196) q[3];
rz(0.4662810131709065) q[3];
ry(1.3674250170767657) q[4];
rz(0.7524581821597085) q[4];
ry(-2.9433339760594888) q[5];
rz(-0.441812995191718) q[5];
ry(0.239635656704817) q[6];
rz(0.1769343590125584) q[6];
ry(-2.3706789702954603) q[7];
rz(-0.20552051212077327) q[7];
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
ry(-2.430930070770064) q[0];
rz(0.5885482853849284) q[0];
ry(0.18836438090941154) q[1];
rz(1.8818025009806743) q[1];
ry(-1.7522477282025601) q[2];
rz(0.7948514531131917) q[2];
ry(-1.6104180580823129) q[3];
rz(-1.5206257140669) q[3];
ry(0.287207322170258) q[4];
rz(-1.68605685505476) q[4];
ry(-0.26944044618021556) q[5];
rz(1.2697310968717437) q[5];
ry(-1.0993779596625055) q[6];
rz(-0.786678375878929) q[6];
ry(-2.6941564474168773) q[7];
rz(-1.2985944351018066) q[7];
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
ry(-2.599295430762217) q[0];
rz(1.0898283440855514) q[0];
ry(0.6452558641619053) q[1];
rz(-1.0405870250203946) q[1];
ry(-0.8946805398685356) q[2];
rz(0.7024342070141217) q[2];
ry(2.4449128508153284) q[3];
rz(0.004965177204170985) q[3];
ry(0.26212055124429323) q[4];
rz(-1.8566667935170553) q[4];
ry(-1.5845899407866149) q[5];
rz(2.2544927061695375) q[5];
ry(-1.0880724066877223) q[6];
rz(1.5491047713324435) q[6];
ry(0.3877138156976683) q[7];
rz(0.4739296475031365) q[7];
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
ry(1.9285227143593127) q[0];
rz(-0.56777978410189) q[0];
ry(-0.3221067776786777) q[1];
rz(0.13837602029822538) q[1];
ry(2.9571054713887257) q[2];
rz(-2.2458336946130277) q[2];
ry(2.5119695575216987) q[3];
rz(-2.2010912755687073) q[3];
ry(-0.6753147797057479) q[4];
rz(-2.335784038828571) q[4];
ry(1.1565012035895328) q[5];
rz(1.523019885475878) q[5];
ry(2.931122237807826) q[6];
rz(1.3239128653283272) q[6];
ry(0.21349595831825094) q[7];
rz(0.5078039120228038) q[7];
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
ry(1.9869886763527744) q[0];
rz(2.04568792187085) q[0];
ry(-2.698288153425152) q[1];
rz(0.28061642668028236) q[1];
ry(0.43053080327587256) q[2];
rz(0.8075700489014367) q[2];
ry(-0.5408187468977266) q[3];
rz(-2.409023868504364) q[3];
ry(1.0575297861976471) q[4];
rz(2.1208093112404036) q[4];
ry(-3.0802038023926883) q[5];
rz(-1.5002770583168918) q[5];
ry(-2.2215489363057825) q[6];
rz(-1.3288614071120026) q[6];
ry(-0.5135782308862655) q[7];
rz(1.1063272720154764) q[7];
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
ry(1.741314630057123) q[0];
rz(0.9998419757163308) q[0];
ry(0.8051767047750861) q[1];
rz(-1.6774215309376943) q[1];
ry(0.10105903656228553) q[2];
rz(0.05032291908389722) q[2];
ry(-1.8548514804641782) q[3];
rz(0.8942068932120685) q[3];
ry(3.0059265785688334) q[4];
rz(3.136446014007726) q[4];
ry(1.3835912950820302) q[5];
rz(1.3749071548671559) q[5];
ry(1.2327484653027927) q[6];
rz(1.4846243327161297) q[6];
ry(0.54055109333181) q[7];
rz(-2.0534539550261544) q[7];
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
ry(2.0967984328151594) q[0];
rz(-0.28799943952364065) q[0];
ry(-0.8855894336933047) q[1];
rz(2.9765541415734824) q[1];
ry(2.864127643773976) q[2];
rz(1.680835828712341) q[2];
ry(0.08290310445751459) q[3];
rz(0.015456847132126585) q[3];
ry(2.3270816314105107) q[4];
rz(-1.400944126325697) q[4];
ry(1.1294516953329286) q[5];
rz(1.3924338833116803) q[5];
ry(0.6051378303041115) q[6];
rz(-2.1771046080322023) q[6];
ry(1.8283093617869985) q[7];
rz(-3.0414148184529552) q[7];
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
ry(-0.9531566904450396) q[0];
rz(2.898204911980813) q[0];
ry(-0.3960621735407628) q[1];
rz(-2.3519314893708696) q[1];
ry(0.5394420166722114) q[2];
rz(1.6236701879830076) q[2];
ry(1.900842310096495) q[3];
rz(-1.6326473962950996) q[3];
ry(-0.2154337033132091) q[4];
rz(3.116428467006147) q[4];
ry(2.192247135919912) q[5];
rz(-0.13972017411000426) q[5];
ry(0.6160745313984981) q[6];
rz(-0.32045878271192946) q[6];
ry(-0.23374369193468247) q[7];
rz(-1.9003953213480642) q[7];
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
ry(1.8245656803531372) q[0];
rz(1.4760070469266644) q[0];
ry(-1.293853623146707) q[1];
rz(1.278618943117317) q[1];
ry(-0.8046904276856459) q[2];
rz(2.967303436278937) q[2];
ry(-3.0424453048388513) q[3];
rz(0.5977565884117029) q[3];
ry(-0.6048540494468853) q[4];
rz(-1.9634489139539824) q[4];
ry(2.216858416927562) q[5];
rz(-0.9750427515067749) q[5];
ry(-1.04570337503344) q[6];
rz(0.3722004621414642) q[6];
ry(-0.9717281486942235) q[7];
rz(2.802844809583517) q[7];
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
ry(-1.8202310056128121) q[0];
rz(-0.10656584554548319) q[0];
ry(1.2216221305728585) q[1];
rz(2.3488685010535746) q[1];
ry(2.185815082120963) q[2];
rz(2.9922972161393173) q[2];
ry(-3.0814826276722345) q[3];
rz(1.855710168486462) q[3];
ry(0.4199503879632829) q[4];
rz(-0.2532592935100402) q[4];
ry(-2.5129180822765753) q[5];
rz(-1.4252364161453421) q[5];
ry(2.040752960380331) q[6];
rz(-2.15075377762882) q[6];
ry(1.5664132892356188) q[7];
rz(2.206252242311244) q[7];
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
ry(-3.112608886812923) q[0];
rz(-1.0962104666376284) q[0];
ry(0.9153615409434082) q[1];
rz(-1.4508013364276167) q[1];
ry(-0.6006988768379046) q[2];
rz(-1.746202885180696) q[2];
ry(1.6972403594208967) q[3];
rz(2.9736782326811713) q[3];
ry(0.6388041394814509) q[4];
rz(1.3226677901197395) q[4];
ry(1.6186607793118455) q[5];
rz(-1.102068267901592) q[5];
ry(1.3945761348270287) q[6];
rz(1.3334184428240952) q[6];
ry(0.42353601855638523) q[7];
rz(-2.5820782420597355) q[7];
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
ry(-2.3595185200564646) q[0];
rz(1.0469650843222862) q[0];
ry(-2.3338116529577744) q[1];
rz(-1.9562975374885931) q[1];
ry(1.691295728664402) q[2];
rz(0.6028269805638328) q[2];
ry(-0.4850976198161543) q[3];
rz(-2.1332183816262047) q[3];
ry(0.66525566812565) q[4];
rz(2.417305185446608) q[4];
ry(2.0827790591843587) q[5];
rz(-1.9533366806370482) q[5];
ry(-0.8220846168397422) q[6];
rz(2.748638511751926) q[6];
ry(-2.0812424824496194) q[7];
rz(1.831222512302338) q[7];
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
ry(1.1697869571794552) q[0];
rz(2.716253149243605) q[0];
ry(0.8508920890396133) q[1];
rz(0.9782388007211487) q[1];
ry(-2.4273902101146394) q[2];
rz(-1.1015444034226975) q[2];
ry(-0.8204215854244561) q[3];
rz(2.2635782909545874) q[3];
ry(0.6870108975004369) q[4];
rz(-2.825249561153331) q[4];
ry(3.037233087502322) q[5];
rz(1.1813571467218562) q[5];
ry(2.229991873882857) q[6];
rz(1.2552182035631363) q[6];
ry(2.47696670793656) q[7];
rz(-1.445511012967319) q[7];
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
ry(-0.3127255814319598) q[0];
rz(-3.0434046930199212) q[0];
ry(-0.6027925231209437) q[1];
rz(-2.262539439664341) q[1];
ry(-0.6140786941133838) q[2];
rz(-0.5442760150919597) q[2];
ry(-0.3087975600333408) q[3];
rz(-1.4173714309007934) q[3];
ry(-0.30337668239202586) q[4];
rz(0.9703928345358213) q[4];
ry(1.6093054178617745) q[5];
rz(0.9666486287953933) q[5];
ry(2.9472574631698083) q[6];
rz(-1.2848389063761152) q[6];
ry(-0.955545228658397) q[7];
rz(-0.04957815473452154) q[7];
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
ry(-1.7808914382564633) q[0];
rz(-0.43605265528081605) q[0];
ry(1.0750964760890502) q[1];
rz(-1.6791861926748455) q[1];
ry(2.285814661502536) q[2];
rz(-0.14088398058564505) q[2];
ry(0.16961380791917938) q[3];
rz(-1.1243454137998206) q[3];
ry(-2.69969524912754) q[4];
rz(-3.133037467214176) q[4];
ry(-1.0398511427598343) q[5];
rz(-1.3060952134170192) q[5];
ry(-2.135809897850761) q[6];
rz(-0.8973871147388572) q[6];
ry(1.9528385496284189) q[7];
rz(0.9413195500302795) q[7];
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
ry(-1.2202332601409847) q[0];
rz(1.3676223850913143) q[0];
ry(-2.484344401688596) q[1];
rz(-0.22844558388232533) q[1];
ry(-0.8053895947139926) q[2];
rz(2.7288620939684356) q[2];
ry(-2.186467082211273) q[3];
rz(2.853582661113085) q[3];
ry(-1.0183732773975351) q[4];
rz(-3.004653049239993) q[4];
ry(-1.7076988529605102) q[5];
rz(-1.69003451353619) q[5];
ry(1.137078374484049) q[6];
rz(0.7084655446783028) q[6];
ry(-2.0606670765954114) q[7];
rz(-1.916087019695442) q[7];
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
ry(3.107245562118563) q[0];
rz(-2.7406832924258553) q[0];
ry(-2.3626530927805205) q[1];
rz(2.501201348670671) q[1];
ry(-0.7182633409900829) q[2];
rz(-2.179563080295212) q[2];
ry(-2.494715760858874) q[3];
rz(2.351019562398687) q[3];
ry(0.9126249632581594) q[4];
rz(-2.1635891790521535) q[4];
ry(-2.036590550456814) q[5];
rz(2.9656557915129866) q[5];
ry(-2.743943153492491) q[6];
rz(2.9306657879120017) q[6];
ry(-0.7312564719940333) q[7];
rz(3.0569108685574906) q[7];
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
ry(2.3474758120205435) q[0];
rz(-0.6325689450893801) q[0];
ry(-1.0836412070024553) q[1];
rz(0.36059454607973207) q[1];
ry(2.3789255142199766) q[2];
rz(1.5253532376060903) q[2];
ry(0.310627261380715) q[3];
rz(3.073109563025428) q[3];
ry(2.190200854604895) q[4];
rz(2.527335576301456) q[4];
ry(-2.3709656900762783) q[5];
rz(0.13266612952151446) q[5];
ry(-1.5225217627459324) q[6];
rz(-1.6121850718178867) q[6];
ry(1.4036460748956363) q[7];
rz(2.2221144909459385) q[7];