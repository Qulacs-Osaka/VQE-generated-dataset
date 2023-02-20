OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.3652652784236681) q[0];
rz(-0.20594750297254993) q[0];
ry(0.1263657482422436) q[1];
rz(0.4900689729981462) q[1];
ry(1.0682451993418702) q[2];
rz(-1.4341212744273586) q[2];
ry(-2.7790548217983013) q[3];
rz(2.6356791909125383) q[3];
ry(-1.7343176621577054) q[4];
rz(0.7762726791385828) q[4];
ry(2.7387787687767577) q[5];
rz(0.8271300449978214) q[5];
ry(-2.8340224214449945) q[6];
rz(0.05677330986000567) q[6];
ry(-0.5421728607591816) q[7];
rz(2.1763961941047336) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.3255434645544777) q[0];
rz(-0.8319311353119616) q[0];
ry(1.9781351064427755) q[1];
rz(1.7671283611112065) q[1];
ry(-2.232046082092123) q[2];
rz(0.2817585994080601) q[2];
ry(-2.096512413276619) q[3];
rz(3.1012175542465257) q[3];
ry(0.6528515431008303) q[4];
rz(0.14196567730268822) q[4];
ry(0.8556295590192121) q[5];
rz(1.2673750078209478) q[5];
ry(2.545410339126432) q[6];
rz(-2.845565188047901) q[6];
ry(-0.4255257585472106) q[7];
rz(-2.2423553098539477) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.2029182692306275) q[0];
rz(1.050655147111904) q[0];
ry(-2.769510193876482) q[1];
rz(-1.4062028109222604) q[1];
ry(1.9853374787941425) q[2];
rz(-1.9964623282334548) q[2];
ry(0.3299341964635698) q[3];
rz(-1.1929835454872393) q[3];
ry(1.700355626323896) q[4];
rz(0.8088067122095808) q[4];
ry(1.9924779137000803) q[5];
rz(0.5100835498456107) q[5];
ry(-0.18641300349886955) q[6];
rz(0.2667117543190225) q[6];
ry(2.7546997780904716) q[7];
rz(1.932111895002628) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.8463979376354382) q[0];
rz(-1.8238354796125826) q[0];
ry(-0.24581934826789165) q[1];
rz(-2.8577042165018187) q[1];
ry(2.9605507681076624) q[2];
rz(0.9816661403241668) q[2];
ry(-1.3657976524494164) q[3];
rz(1.2897452113385257) q[3];
ry(1.558848035856329) q[4];
rz(-2.6843746531190162) q[4];
ry(1.6980025521460627) q[5];
rz(0.11191481714930457) q[5];
ry(2.7608746000607596) q[6];
rz(2.4380190083518642) q[6];
ry(-2.4681691002464907) q[7];
rz(0.38286908230438493) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.167436268414516) q[0];
rz(-1.4731489504835413) q[0];
ry(1.8710280701984336) q[1];
rz(2.407291425736665) q[1];
ry(0.5931734438613825) q[2];
rz(0.8865376999951158) q[2];
ry(-1.1617720358485402) q[3];
rz(-1.2832255134264092) q[3];
ry(0.9934951708028236) q[4];
rz(0.16695154954344638) q[4];
ry(1.7199682018825833) q[5];
rz(1.5589949947169186) q[5];
ry(1.0366045482540098) q[6];
rz(2.1705957796652933) q[6];
ry(2.217728869670493) q[7];
rz(1.022485086662499) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.2349522362404373) q[0];
rz(1.017802796992691) q[0];
ry(1.2898366887094417) q[1];
rz(-2.9569035642497346) q[1];
ry(2.105348942107058) q[2];
rz(-1.8385129776751175) q[2];
ry(0.5296075364000711) q[3];
rz(0.8671895546999308) q[3];
ry(-1.6203142443341887) q[4];
rz(1.0004536002358488) q[4];
ry(2.8218108454375974) q[5];
rz(-1.0151533674197883) q[5];
ry(0.30114041344039316) q[6];
rz(1.8864630938003657) q[6];
ry(-0.6098157438338383) q[7];
rz(-2.2369276205381254) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.23415903806975) q[0];
rz(-1.3250821767481513) q[0];
ry(0.34495244559576405) q[1];
rz(0.6291712093208034) q[1];
ry(1.0817036651615055) q[2];
rz(-1.5100884393138108) q[2];
ry(-0.21641992294377732) q[3];
rz(-1.8349372821213628) q[3];
ry(0.6424394080255089) q[4];
rz(-2.673643029443232) q[4];
ry(1.4216106045643295) q[5];
rz(-0.4010163934328555) q[5];
ry(-0.8717071282835338) q[6];
rz(-2.856679389945065) q[6];
ry(0.7222512149674069) q[7];
rz(1.4162376317402563) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.3500072293217238) q[0];
rz(0.5324049946264554) q[0];
ry(-0.9884601340851704) q[1];
rz(-1.0082481506583154) q[1];
ry(3.0707208188640784) q[2];
rz(0.4252502542883198) q[2];
ry(2.3150120672226513) q[3];
rz(-2.8663690036881384) q[3];
ry(-2.722141565549554) q[4];
rz(-0.9634148873297743) q[4];
ry(1.2202151937338448) q[5];
rz(-1.1389874910552105) q[5];
ry(3.0069463092412785) q[6];
rz(-2.8054415177519156) q[6];
ry(-1.836746646138201) q[7];
rz(-0.6821215062944299) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.4957320455439307) q[0];
rz(0.04676192356004183) q[0];
ry(-0.9914741886650154) q[1];
rz(1.2425044716864535) q[1];
ry(-0.4995931779428098) q[2];
rz(-1.7861927866845286) q[2];
ry(1.150175367008714) q[3];
rz(1.1092511415142605) q[3];
ry(2.228549161868991) q[4];
rz(2.5846283862722403) q[4];
ry(0.7220248124155235) q[5];
rz(2.1958530567155687) q[5];
ry(-2.6025238452909663) q[6];
rz(-0.5880490661991037) q[6];
ry(-1.7952215002201224) q[7];
rz(-0.7564119057618441) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.3137047157276758) q[0];
rz(-0.6706013044061939) q[0];
ry(0.30984760397540967) q[1];
rz(0.942945142692599) q[1];
ry(0.9320409843173003) q[2];
rz(0.5876730691356905) q[2];
ry(-0.6005165411230635) q[3];
rz(0.3941976097559427) q[3];
ry(2.166705858538563) q[4];
rz(0.6746762192362084) q[4];
ry(-1.7215542720454995) q[5];
rz(-0.6452723442412484) q[5];
ry(2.9490564561815584) q[6];
rz(2.5550394519674495) q[6];
ry(-1.7489386886742255) q[7];
rz(-0.4631108809443143) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.5079612944945877) q[0];
rz(-1.241830316463888) q[0];
ry(0.342000703336737) q[1];
rz(2.291066755411297) q[1];
ry(1.9075965199291542) q[2];
rz(-2.5370756561410452) q[2];
ry(1.415657047448603) q[3];
rz(-2.0741300834214433) q[3];
ry(2.663313313838793) q[4];
rz(-1.218090357282007) q[4];
ry(2.4969562633848925) q[5];
rz(-1.6283927394150046) q[5];
ry(1.9153937803028438) q[6];
rz(-1.4362327758521203) q[6];
ry(-2.780354898429796) q[7];
rz(-0.6761257431410487) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.467470663167012) q[0];
rz(0.8915588233615015) q[0];
ry(1.9038724483260756) q[1];
rz(0.9743757949381375) q[1];
ry(1.4514468733874157) q[2];
rz(-1.7656750113680948) q[2];
ry(-0.7949029949025399) q[3];
rz(-0.30847875006656816) q[3];
ry(1.3247207918179196) q[4];
rz(-2.558101401059401) q[4];
ry(1.042807991869796) q[5];
rz(2.4559512232246927) q[5];
ry(-0.8372398702543897) q[6];
rz(-1.251072671757333) q[6];
ry(0.2728030204544022) q[7];
rz(-1.3157736367120867) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.4297562534379863) q[0];
rz(2.427988231629865) q[0];
ry(-1.6371939542586997) q[1];
rz(-1.4866731839305611) q[1];
ry(-1.4147703875392699) q[2];
rz(-2.626008803403804) q[2];
ry(-1.5894361733078712) q[3];
rz(-0.33916197928993774) q[3];
ry(-2.950511807964591) q[4];
rz(-0.9238815896799011) q[4];
ry(-1.081370932435146) q[5];
rz(-2.4153536867834737) q[5];
ry(1.514974766950518) q[6];
rz(0.9253460067084838) q[6];
ry(1.8253255892122615) q[7];
rz(2.6694379166871864) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.485028712068283) q[0];
rz(1.6287545276816895) q[0];
ry(1.4179671077267149) q[1];
rz(-3.009653917215934) q[1];
ry(-1.510914925873773) q[2];
rz(0.760260970566289) q[2];
ry(-1.8865275353553743) q[3];
rz(-2.7216656990238226) q[3];
ry(0.8208796501616231) q[4];
rz(-1.8440861971637799) q[4];
ry(1.9047773150907712) q[5];
rz(-2.2193664704415745) q[5];
ry(-2.2182071629629507) q[6];
rz(0.9420698911306848) q[6];
ry(1.5812359425150422) q[7];
rz(-1.7213214235172791) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.6992847423229325) q[0];
rz(0.08518582683615339) q[0];
ry(-0.927734076481441) q[1];
rz(1.0364281185583213) q[1];
ry(1.9369275840496307) q[2];
rz(-2.2353525937630874) q[2];
ry(2.934873344650423) q[3];
rz(2.8230028901760242) q[3];
ry(0.5919462448242792) q[4];
rz(2.330239942751234) q[4];
ry(-2.946628362677617) q[5];
rz(-1.2497429812984313) q[5];
ry(-0.614648180111675) q[6];
rz(0.827946464727989) q[6];
ry(0.8346093763933151) q[7];
rz(-1.2482047997423287) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.2977938094997463) q[0];
rz(1.6485632459744597) q[0];
ry(0.8978714569842552) q[1];
rz(-0.05552234261016908) q[1];
ry(2.8600517153977507) q[2];
rz(-0.3192398740423572) q[2];
ry(1.784296065138589) q[3];
rz(0.13422611567770293) q[3];
ry(1.993469472296896) q[4];
rz(-0.9943743580522643) q[4];
ry(0.6145353391713781) q[5];
rz(1.0405246382152573) q[5];
ry(-2.0445028021252494) q[6];
rz(2.6477918339565103) q[6];
ry(-1.3723923134712912) q[7];
rz(-1.0113417515966476) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.0542791459827614) q[0];
rz(-2.562379570599078) q[0];
ry(-0.4600446469484862) q[1];
rz(-0.5685961289363607) q[1];
ry(0.24291693493611177) q[2];
rz(2.2651213342689704) q[2];
ry(-0.5528117898955287) q[3];
rz(-0.19690733469553032) q[3];
ry(0.043113917475036345) q[4];
rz(1.3285231723455349) q[4];
ry(1.1261432710327195) q[5];
rz(-1.5299107842720314) q[5];
ry(-0.9863402514553963) q[6];
rz(1.5446188155361544) q[6];
ry(-0.19855685235105602) q[7];
rz(-2.978036254094121) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.8676573954711682) q[0];
rz(2.645954727031434) q[0];
ry(-1.1721049405013002) q[1];
rz(-2.708420445145625) q[1];
ry(-1.5708204234794285) q[2];
rz(-1.6240921208166157) q[2];
ry(-1.7012075099094188) q[3];
rz(0.5795546836036483) q[3];
ry(2.344147484497577) q[4];
rz(-1.7780438017952132) q[4];
ry(1.4894016200941982) q[5];
rz(-1.347495592921073) q[5];
ry(1.1885479688078249) q[6];
rz(-2.3679975045353987) q[6];
ry(1.892557258268031) q[7];
rz(-0.8318493185874019) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.1447259752183383) q[0];
rz(0.415449616105108) q[0];
ry(1.7331025031459077) q[1];
rz(-1.259122162951777) q[1];
ry(-0.8074662243898069) q[2];
rz(-1.0618640898198182) q[2];
ry(2.787097336970947) q[3];
rz(2.4007330194658296) q[3];
ry(-0.6917820444881432) q[4];
rz(0.8639008043974252) q[4];
ry(1.2517837099904618) q[5];
rz(-2.297861413382488) q[5];
ry(-0.9459414699505428) q[6];
rz(1.775854816318591) q[6];
ry(-2.7764437243937152) q[7];
rz(-1.5206973409803552) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.1692625292389975) q[0];
rz(0.6778168504614333) q[0];
ry(-0.564689063839503) q[1];
rz(-2.1849753353651096) q[1];
ry(2.681629621476353) q[2];
rz(-1.6271986288996079) q[2];
ry(-0.4358344988091139) q[3];
rz(2.773013392165103) q[3];
ry(0.7226077699099267) q[4];
rz(0.3852080976942487) q[4];
ry(-0.9487755565291129) q[5];
rz(0.8919214664383827) q[5];
ry(-1.9109705432438595) q[6];
rz(0.8583809719383851) q[6];
ry(-2.814529718383657) q[7];
rz(2.555891708572969) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.1544773302690308) q[0];
rz(2.807401575480839) q[0];
ry(-2.983957942724712) q[1];
rz(0.300070347477576) q[1];
ry(-0.9088441565467478) q[2];
rz(1.142572449038345) q[2];
ry(0.7799081976966602) q[3];
rz(3.0653986774908075) q[3];
ry(2.344607782393977) q[4];
rz(0.09237508849349747) q[4];
ry(3.1156876074051922) q[5];
rz(3.0275605865321333) q[5];
ry(-2.9048739603955966) q[6];
rz(-0.8884602858832379) q[6];
ry(2.824285119606831) q[7];
rz(-0.9308370304784055) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.0132047974008458) q[0];
rz(1.2655613451485488) q[0];
ry(1.7519909633864348) q[1];
rz(2.98283166857814) q[1];
ry(-0.6983143320130696) q[2];
rz(1.2425672169624207) q[2];
ry(1.4576875446896087) q[3];
rz(2.6045936709416315) q[3];
ry(2.582741783107388) q[4];
rz(-0.9242689830029516) q[4];
ry(2.929516572668763) q[5];
rz(-1.7621287806135433) q[5];
ry(0.6637674540007782) q[6];
rz(2.939251276705483) q[6];
ry(-1.330409780876084) q[7];
rz(0.7624555923660612) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.7693078206631183) q[0];
rz(0.48717810222377034) q[0];
ry(0.6926716716301858) q[1];
rz(-0.39996774620780073) q[1];
ry(2.816710219567094) q[2];
rz(-0.9113684771392424) q[2];
ry(2.0653424904703885) q[3];
rz(-3.0074408222883493) q[3];
ry(0.8514793801578647) q[4];
rz(-3.1337887435204625) q[4];
ry(-0.8347630318585866) q[5];
rz(2.1022695603547534) q[5];
ry(2.3585495924229067) q[6];
rz(3.087696370623412) q[6];
ry(-0.9582040960933813) q[7];
rz(0.8313393512420868) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.5594194927354756) q[0];
rz(2.0625235123430534) q[0];
ry(-1.6446343025512224) q[1];
rz(0.46352283755816537) q[1];
ry(1.1124691610928643) q[2];
rz(1.0423465583194982) q[2];
ry(-1.54357610850119) q[3];
rz(-0.6783815366601234) q[3];
ry(2.9686563699936186) q[4];
rz(2.725178007446607) q[4];
ry(1.7955039367394754) q[5];
rz(-1.72523643775884) q[5];
ry(-0.8326236514628698) q[6];
rz(0.7706637301578639) q[6];
ry(-0.23265439459552614) q[7];
rz(0.7156588637777981) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.39510004770499796) q[0];
rz(0.4996013259378503) q[0];
ry(-0.3303078900797078) q[1];
rz(2.9036133191366034) q[1];
ry(-2.5000405724023893) q[2];
rz(-2.5479744839213216) q[2];
ry(-1.0940521316026723) q[3];
rz(-1.7457758791405666) q[3];
ry(-0.5288976966958656) q[4];
rz(-1.3756654761715104) q[4];
ry(-0.836663008055246) q[5];
rz(-1.2284548831417472) q[5];
ry(2.0397076088312094) q[6];
rz(-2.6624409918064513) q[6];
ry(-0.5002845054947223) q[7];
rz(-1.6682449046622176) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.4544129034513453) q[0];
rz(-0.8915525413400613) q[0];
ry(1.4550426071294735) q[1];
rz(1.9414228797055173) q[1];
ry(-1.7558418988807007) q[2];
rz(-2.8640219101677666) q[2];
ry(0.22275011136863923) q[3];
rz(3.0857996865810553) q[3];
ry(1.007848036010981) q[4];
rz(-1.2868787544422702) q[4];
ry(-1.7111216515536314) q[5];
rz(1.5144240146503947) q[5];
ry(1.920821727388935) q[6];
rz(2.40683844525096) q[6];
ry(-3.0450597877993197) q[7];
rz(-1.5937566734895794) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(2.116782222327742) q[0];
rz(-2.458728956518548) q[0];
ry(1.822507615892552) q[1];
rz(-0.5652932880677717) q[1];
ry(0.3381780409420565) q[2];
rz(-1.3154770701278873) q[2];
ry(1.1485872553585867) q[3];
rz(-1.1205850346570998) q[3];
ry(-2.137195816904806) q[4];
rz(-1.905924554774514) q[4];
ry(-2.018498224789191) q[5];
rz(1.508593304499608) q[5];
ry(-0.8976203670662165) q[6];
rz(0.6915590835548154) q[6];
ry(-0.4789826459708601) q[7];
rz(-1.3337940976775213) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-2.2359277387691394) q[0];
rz(1.6295568337175181) q[0];
ry(2.6894321219460173) q[1];
rz(-1.9043073054155217) q[1];
ry(0.2563844391428059) q[2];
rz(2.222282991913481) q[2];
ry(-0.0026279077424424275) q[3];
rz(0.5645296860538783) q[3];
ry(1.6516264080040974) q[4];
rz(-0.5536315185855428) q[4];
ry(-1.2192740448808934) q[5];
rz(-2.0584339347951603) q[5];
ry(2.747483633435761) q[6];
rz(0.8370454621447303) q[6];
ry(2.007788578744356) q[7];
rz(-2.9081123621166816) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-1.2366111607234989) q[0];
rz(-1.4378165242703762) q[0];
ry(-0.7159402371397716) q[1];
rz(-0.3108893971218061) q[1];
ry(-2.7475393940763375) q[2];
rz(-0.16219701149676438) q[2];
ry(-2.254326214848888) q[3];
rz(-0.582986782758586) q[3];
ry(0.5873174262953831) q[4];
rz(-2.87910987422676) q[4];
ry(0.6346844327148714) q[5];
rz(1.7415875313140485) q[5];
ry(0.7649377224848085) q[6];
rz(0.09370979751543423) q[6];
ry(-1.621362478836652) q[7];
rz(2.965943645059147) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(-0.5358700641745813) q[0];
rz(-0.7289741289639017) q[0];
ry(-1.8439086759140102) q[1];
rz(0.2226409174385099) q[1];
ry(-1.6975053080497386) q[2];
rz(2.904227592056325) q[2];
ry(-0.23838135009966965) q[3];
rz(-0.5240337116480935) q[3];
ry(2.972638905930869) q[4];
rz(0.06983630812074058) q[4];
ry(-0.7136471974240267) q[5];
rz(1.2165548577880259) q[5];
ry(0.8126451362169123) q[6];
rz(1.7047212356983281) q[6];
ry(-1.9568989740231029) q[7];
rz(0.8620290904393818) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.3299787864682546) q[0];
rz(1.0947704044853381) q[0];
ry(2.1643064108917667) q[1];
rz(-0.46808866988727615) q[1];
ry(2.809964831423918) q[2];
rz(0.7762460208006645) q[2];
ry(3.0549244682049372) q[3];
rz(-2.3677436028335563) q[3];
ry(1.132124920498283) q[4];
rz(0.4544171857927892) q[4];
ry(-2.093524496341095) q[5];
rz(1.3015710788937884) q[5];
ry(-1.861030548512647) q[6];
rz(-2.639266100903229) q[6];
ry(0.6780945839116153) q[7];
rz(2.381004118419261) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(1.4696848410996355) q[0];
rz(2.2157226512837913) q[0];
ry(-0.3395016448696521) q[1];
rz(-1.1173062383783057) q[1];
ry(0.4115369203588712) q[2];
rz(-1.8352618146596915) q[2];
ry(-1.8104162743954175) q[3];
rz(-2.247383546100819) q[3];
ry(2.3630277464490774) q[4];
rz(-0.5756042623709984) q[4];
ry(2.6165580654048624) q[5];
rz(2.4618206394856963) q[5];
ry(-2.6028847025022435) q[6];
rz(-0.44627383232150125) q[6];
ry(-0.495340280352422) q[7];
rz(-2.701789814999076) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
ry(0.3324763848342059) q[0];
rz(0.7642785591728014) q[0];
ry(-2.848602651315659) q[1];
rz(2.1876535004819426) q[1];
ry(-3.1042337358671688) q[2];
rz(-1.4646390870388544) q[2];
ry(0.025083761636257584) q[3];
rz(-3.0908824359598324) q[3];
ry(2.5039600123706633) q[4];
rz(1.0279070490381774) q[4];
ry(2.0924288013105152) q[5];
rz(-2.8061882521657093) q[5];
ry(-0.5935491500907097) q[6];
rz(0.33026883077152025) q[6];
ry(-0.48305249859683475) q[7];
rz(2.407129082323818) q[7];