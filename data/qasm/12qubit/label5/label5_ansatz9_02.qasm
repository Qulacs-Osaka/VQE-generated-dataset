OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.7443285188788593) q[0];
ry(1.0765132414734913) q[1];
cx q[0],q[1];
ry(1.442477720995365) q[0];
ry(2.6563420616860736) q[1];
cx q[0],q[1];
ry(3.1366241092133267) q[2];
ry(-0.45040569152640847) q[3];
cx q[2],q[3];
ry(0.3907503401331409) q[2];
ry(-0.637655602470144) q[3];
cx q[2],q[3];
ry(2.552426211068891) q[4];
ry(1.622613528632508) q[5];
cx q[4],q[5];
ry(-2.6398779249406616) q[4];
ry(1.742681659664906) q[5];
cx q[4],q[5];
ry(3.113116206241825) q[6];
ry(2.6378962039935043) q[7];
cx q[6],q[7];
ry(-0.4442110132336019) q[6];
ry(2.168830232807294) q[7];
cx q[6],q[7];
ry(-2.237255063613472) q[8];
ry(0.5218055987767901) q[9];
cx q[8],q[9];
ry(2.7429854339055746) q[8];
ry(0.5170148115049189) q[9];
cx q[8],q[9];
ry(-1.048228645466617) q[10];
ry(-0.36427564809239815) q[11];
cx q[10],q[11];
ry(-1.2784447762238473) q[10];
ry(-1.6518315648833335) q[11];
cx q[10],q[11];
ry(-1.53792667981587) q[0];
ry(0.49558787408780525) q[2];
cx q[0],q[2];
ry(3.0999936992231842) q[0];
ry(2.321304252023837) q[2];
cx q[0],q[2];
ry(-0.5947493356779432) q[2];
ry(0.6114669788830228) q[4];
cx q[2],q[4];
ry(-1.281239810619372) q[2];
ry(2.1980084862033666) q[4];
cx q[2],q[4];
ry(0.12980684622452815) q[4];
ry(-1.0801814673824435) q[6];
cx q[4],q[6];
ry(-2.7106495821865524) q[4];
ry(-3.0628172819538557) q[6];
cx q[4],q[6];
ry(-2.573760377600367) q[6];
ry(0.7826875460940396) q[8];
cx q[6],q[8];
ry(2.4676607186603277) q[6];
ry(3.1281308807972943) q[8];
cx q[6],q[8];
ry(-0.5575609778981822) q[8];
ry(0.930985081514689) q[10];
cx q[8],q[10];
ry(-0.1481330527492188) q[8];
ry(3.0530826838327747) q[10];
cx q[8],q[10];
ry(3.1223116297856173) q[1];
ry(-0.07336148486724293) q[3];
cx q[1],q[3];
ry(0.8750999843288421) q[1];
ry(-1.297800198895307) q[3];
cx q[1],q[3];
ry(0.44785947187520136) q[3];
ry(2.95377410069388) q[5];
cx q[3],q[5];
ry(1.0954296961596404) q[3];
ry(-1.5565645731410875) q[5];
cx q[3],q[5];
ry(0.4410017012484948) q[5];
ry(2.6100562363684685) q[7];
cx q[5],q[7];
ry(2.9009729384430427) q[5];
ry(2.8791500598928232) q[7];
cx q[5],q[7];
ry(0.0602577625385432) q[7];
ry(2.576755157674517) q[9];
cx q[7],q[9];
ry(-3.125867007574955) q[7];
ry(-3.127244150532053) q[9];
cx q[7],q[9];
ry(-1.707081507444127) q[9];
ry(2.86333640980823) q[11];
cx q[9],q[11];
ry(-2.6865592893801145) q[9];
ry(-0.5460372845543429) q[11];
cx q[9],q[11];
ry(0.8802795296751237) q[0];
ry(-0.4374420722214791) q[3];
cx q[0],q[3];
ry(2.3080068413953763) q[0];
ry(-0.26570965147309167) q[3];
cx q[0],q[3];
ry(1.6446145711955453) q[1];
ry(-0.4481661311466274) q[2];
cx q[1],q[2];
ry(3.079155811013523) q[1];
ry(-2.0857857195401905) q[2];
cx q[1],q[2];
ry(-1.5335766790431116) q[2];
ry(-2.9216744675372) q[5];
cx q[2],q[5];
ry(-1.4993096716712415) q[2];
ry(2.7579163955999824) q[5];
cx q[2],q[5];
ry(1.7064332066100025) q[3];
ry(-1.0484265540739761) q[4];
cx q[3],q[4];
ry(-1.9561725140589772) q[3];
ry(-2.3695980499175615) q[4];
cx q[3],q[4];
ry(-0.8637727135683941) q[4];
ry(-1.4553026658380834) q[7];
cx q[4],q[7];
ry(-2.976743058597328) q[4];
ry(-0.7337110471725273) q[7];
cx q[4],q[7];
ry(-0.3030404181321398) q[5];
ry(1.1893965962794688) q[6];
cx q[5],q[6];
ry(-0.0469680878573806) q[5];
ry(2.9164853521246275) q[6];
cx q[5],q[6];
ry(1.888885590417052) q[6];
ry(-0.6881681700649036) q[9];
cx q[6],q[9];
ry(2.9518856633104535) q[6];
ry(-0.0285805464772535) q[9];
cx q[6],q[9];
ry(2.344484329829306) q[7];
ry(2.489562982284428) q[8];
cx q[7],q[8];
ry(2.792560070040448) q[7];
ry(3.133391557440768) q[8];
cx q[7],q[8];
ry(-0.9448587318606719) q[8];
ry(0.8884429469930446) q[11];
cx q[8],q[11];
ry(-2.645314597451995) q[8];
ry(-0.9568721553994125) q[11];
cx q[8],q[11];
ry(-2.472469945823069) q[9];
ry(-0.8814073854506601) q[10];
cx q[9],q[10];
ry(2.9196091232786303) q[9];
ry(-0.13504711772478467) q[10];
cx q[9],q[10];
ry(2.6608667501230614) q[0];
ry(-0.5239548051658673) q[1];
cx q[0],q[1];
ry(0.7295751779970843) q[0];
ry(-0.5400636181745382) q[1];
cx q[0],q[1];
ry(-0.3705057030654334) q[2];
ry(0.46228646374200727) q[3];
cx q[2],q[3];
ry(-2.1477447703909838) q[2];
ry(-1.2844475529181576) q[3];
cx q[2],q[3];
ry(-0.1737678563122742) q[4];
ry(1.4350766037409677) q[5];
cx q[4],q[5];
ry(2.6762844089398152) q[4];
ry(-1.9965896437254855) q[5];
cx q[4],q[5];
ry(2.922794119363603) q[6];
ry(-0.09606104682761085) q[7];
cx q[6],q[7];
ry(1.7396499928723559) q[6];
ry(-2.5818287415198853) q[7];
cx q[6],q[7];
ry(2.476171950638088) q[8];
ry(-0.48860425792875006) q[9];
cx q[8],q[9];
ry(-2.8367870323468827) q[8];
ry(-2.1227524248977336) q[9];
cx q[8],q[9];
ry(-1.255325430143352) q[10];
ry(0.7770253899695652) q[11];
cx q[10],q[11];
ry(-2.7632894818947316) q[10];
ry(-0.36933300448945394) q[11];
cx q[10],q[11];
ry(-2.710471487940324) q[0];
ry(-0.36136775936699195) q[2];
cx q[0],q[2];
ry(1.402638010594351) q[0];
ry(2.0250726386798714) q[2];
cx q[0],q[2];
ry(1.4069684877992694) q[2];
ry(-0.9992131640288493) q[4];
cx q[2],q[4];
ry(1.554060343795567) q[2];
ry(2.4510962275314263) q[4];
cx q[2],q[4];
ry(-1.2702716837054941) q[4];
ry(-0.31055948225576113) q[6];
cx q[4],q[6];
ry(-0.16845499486309468) q[4];
ry(-2.86347702438036) q[6];
cx q[4],q[6];
ry(-1.640882639207188) q[6];
ry(-1.7395977604844317) q[8];
cx q[6],q[8];
ry(1.5622247172248578) q[6];
ry(3.1412445963736593) q[8];
cx q[6],q[8];
ry(3.0850018782758615) q[8];
ry(2.2358804519020987) q[10];
cx q[8],q[10];
ry(-0.6578031793918182) q[8];
ry(2.2971495466757297) q[10];
cx q[8],q[10];
ry(1.1597300249226374) q[1];
ry(-1.3361293363324247) q[3];
cx q[1],q[3];
ry(0.034475194274412146) q[1];
ry(0.4799252886424055) q[3];
cx q[1],q[3];
ry(3.028283129921054) q[3];
ry(1.7755284254802817) q[5];
cx q[3],q[5];
ry(1.6553738022966646) q[3];
ry(0.715562427036984) q[5];
cx q[3],q[5];
ry(2.2734913001901216) q[5];
ry(2.967261927476375) q[7];
cx q[5],q[7];
ry(-1.6098365475135832) q[5];
ry(-2.847993239873187) q[7];
cx q[5],q[7];
ry(1.6931037257356865) q[7];
ry(-1.7239827217690529) q[9];
cx q[7],q[9];
ry(-1.7823136894579648) q[7];
ry(-0.02307515698704776) q[9];
cx q[7],q[9];
ry(1.9941823302659791) q[9];
ry(2.775189809487358) q[11];
cx q[9],q[11];
ry(-2.64220496998835) q[9];
ry(-1.828372477111923) q[11];
cx q[9],q[11];
ry(0.7394507932846971) q[0];
ry(-0.040953990279002817) q[3];
cx q[0],q[3];
ry(0.8406181032242026) q[0];
ry(-1.1592447090825573) q[3];
cx q[0],q[3];
ry(0.014497517213242439) q[1];
ry(0.046606378997992866) q[2];
cx q[1],q[2];
ry(0.1717771840404989) q[1];
ry(0.9537813223717831) q[2];
cx q[1],q[2];
ry(2.1142272458884026) q[2];
ry(1.2791780063949767) q[5];
cx q[2],q[5];
ry(-1.2223335553607768) q[2];
ry(2.895576150891736) q[5];
cx q[2],q[5];
ry(-0.444797887103294) q[3];
ry(2.4296262319821693) q[4];
cx q[3],q[4];
ry(-0.30163123830854843) q[3];
ry(0.34724399566307007) q[4];
cx q[3],q[4];
ry(-0.6569153500632735) q[4];
ry(1.8147943357970997) q[7];
cx q[4],q[7];
ry(0.6507818053984556) q[4];
ry(0.7492119680770255) q[7];
cx q[4],q[7];
ry(2.1893822480090552) q[5];
ry(1.6111568159536631) q[6];
cx q[5],q[6];
ry(0.3651289505867891) q[5];
ry(-0.34042341247664676) q[6];
cx q[5],q[6];
ry(-1.5360715554310875) q[6];
ry(-1.2397205494531118) q[9];
cx q[6],q[9];
ry(-3.126112521779476) q[6];
ry(0.7685813569675063) q[9];
cx q[6],q[9];
ry(2.331502381518829) q[7];
ry(-1.2363891167659888) q[8];
cx q[7],q[8];
ry(-0.004176893598280758) q[7];
ry(3.1304004707422792) q[8];
cx q[7],q[8];
ry(0.5316634737719169) q[8];
ry(1.8680996947082713) q[11];
cx q[8],q[11];
ry(3.1168283317692267) q[8];
ry(-3.1365892658793895) q[11];
cx q[8],q[11];
ry(-2.2509620045714422) q[9];
ry(-2.0224498575915497) q[10];
cx q[9],q[10];
ry(1.571803338128815) q[9];
ry(-1.5703929234157643) q[10];
cx q[9],q[10];
ry(3.1301619321242153) q[0];
ry(-1.3400785181068358) q[1];
cx q[0],q[1];
ry(2.0655728066088255) q[0];
ry(-2.304990575502697) q[1];
cx q[0],q[1];
ry(1.2827360049343424) q[2];
ry(0.8957986034931278) q[3];
cx q[2],q[3];
ry(-0.8697027587965387) q[2];
ry(2.1012491441171814) q[3];
cx q[2],q[3];
ry(2.018040917672787) q[4];
ry(2.8856796756947616) q[5];
cx q[4],q[5];
ry(2.9922011343440005) q[4];
ry(-0.4856225963604366) q[5];
cx q[4],q[5];
ry(3.047925420995778) q[6];
ry(2.692839362294088) q[7];
cx q[6],q[7];
ry(0.012322097695928846) q[6];
ry(-0.184399747205819) q[7];
cx q[6],q[7];
ry(0.5706611845294125) q[8];
ry(2.647515877228462) q[9];
cx q[8],q[9];
ry(-3.0518171665247693) q[8];
ry(-1.6123375858250066) q[9];
cx q[8],q[9];
ry(-0.9220218565172171) q[10];
ry(-0.26860249921582685) q[11];
cx q[10],q[11];
ry(-1.6186759171853293) q[10];
ry(-3.130912081479099) q[11];
cx q[10],q[11];
ry(1.881295713392067) q[0];
ry(1.403718713931534) q[2];
cx q[0],q[2];
ry(2.9689865078417355) q[0];
ry(-2.5706988147735346) q[2];
cx q[0],q[2];
ry(1.5689073084382121) q[2];
ry(-2.437002727244722) q[4];
cx q[2],q[4];
ry(1.5752294601560415) q[2];
ry(-2.590885324798911) q[4];
cx q[2],q[4];
ry(2.7030749283454747) q[4];
ry(-1.4774402857507707) q[6];
cx q[4],q[6];
ry(-1.1570118968302445) q[4];
ry(-1.5713057131435766) q[6];
cx q[4],q[6];
ry(0.9789899683677811) q[6];
ry(1.5730080979856265) q[8];
cx q[6],q[8];
ry(-1.5688627899925427) q[6];
ry(3.139967283834171) q[8];
cx q[6],q[8];
ry(-2.114349390158102) q[8];
ry(2.164469743075508) q[10];
cx q[8],q[10];
ry(-0.9684613716017719) q[8];
ry(-0.0038260362124786186) q[10];
cx q[8],q[10];
ry(1.0278816566094537) q[1];
ry(-1.9216532211232096) q[3];
cx q[1],q[3];
ry(-0.6913701257734204) q[1];
ry(-1.0223572175258344) q[3];
cx q[1],q[3];
ry(1.840799929399033) q[3];
ry(0.9489512342441837) q[5];
cx q[3],q[5];
ry(-1.2245952849652415) q[3];
ry(1.7622130404550145) q[5];
cx q[3],q[5];
ry(-2.350296256075051) q[5];
ry(0.034682843350764575) q[7];
cx q[5],q[7];
ry(-1.232978226385998) q[5];
ry(-2.8556246917492727) q[7];
cx q[5],q[7];
ry(-0.7185653067896887) q[7];
ry(-0.35803262983912804) q[9];
cx q[7],q[9];
ry(-0.0019713958482228122) q[7];
ry(0.006017308559092882) q[9];
cx q[7],q[9];
ry(-0.23109987167337173) q[9];
ry(2.0143793153551464) q[11];
cx q[9],q[11];
ry(0.40800744044288884) q[9];
ry(-0.14249156763360116) q[11];
cx q[9],q[11];
ry(0.8543712167210533) q[0];
ry(-0.2777968997951765) q[3];
cx q[0],q[3];
ry(2.0081599607589515) q[0];
ry(0.542946311146963) q[3];
cx q[0],q[3];
ry(1.6163100631710452) q[1];
ry(-1.8869230211258363) q[2];
cx q[1],q[2];
ry(-0.5705770384116171) q[1];
ry(1.5035777727641042) q[2];
cx q[1],q[2];
ry(2.269674909476416) q[2];
ry(2.7128471543613957) q[5];
cx q[2],q[5];
ry(1.7249409464415333) q[2];
ry(1.5533092293890114) q[5];
cx q[2],q[5];
ry(-1.372898845758741) q[3];
ry(1.3043302579884788) q[4];
cx q[3],q[4];
ry(-0.02090544750745238) q[3];
ry(-0.0015119557885912016) q[4];
cx q[3],q[4];
ry(0.25789403925676174) q[4];
ry(2.1343564197780838) q[7];
cx q[4],q[7];
ry(-1.5613110878428602) q[4];
ry(1.567237285206461) q[7];
cx q[4],q[7];
ry(2.6991955154400666) q[5];
ry(1.6763512225246056) q[6];
cx q[5],q[6];
ry(0.001408784289073317) q[5];
ry(3.1412412687184386) q[6];
cx q[5],q[6];
ry(-2.3323267605655955) q[6];
ry(-1.6570867311874435) q[9];
cx q[6],q[9];
ry(1.5617821423787062) q[6];
ry(-0.007674840968909358) q[9];
cx q[6],q[9];
ry(1.5701366284956686) q[7];
ry(-0.5644349695922266) q[8];
cx q[7],q[8];
ry(-3.122947381773019) q[7];
ry(-1.5746972086541715) q[8];
cx q[7],q[8];
ry(-1.094636109854636) q[8];
ry(1.0955324817750132) q[11];
cx q[8],q[11];
ry(-2.0804171346176856) q[8];
ry(1.5714819023853093) q[11];
cx q[8],q[11];
ry(0.6722594966994002) q[9];
ry(-3.047759619331466) q[10];
cx q[9],q[10];
ry(1.5456093955668315) q[9];
ry(3.1394423840921846) q[10];
cx q[9],q[10];
ry(-1.8224319650453324) q[0];
ry(-2.693169749337091) q[1];
cx q[0],q[1];
ry(1.2080170095221328) q[0];
ry(-2.264513436521571) q[1];
cx q[0],q[1];
ry(-2.6797373396777515) q[2];
ry(-0.4692398070730954) q[3];
cx q[2],q[3];
ry(0.475109162786632) q[2];
ry(-0.24974157388337392) q[3];
cx q[2],q[3];
ry(-0.08443290384091107) q[4];
ry(1.564944955085785) q[5];
cx q[4],q[5];
ry(0.8891619275242313) q[4];
ry(1.0506222335274167) q[5];
cx q[4],q[5];
ry(-0.9201017814631172) q[6];
ry(-3.1177648669663) q[7];
cx q[6],q[7];
ry(-1.5310132879889158) q[6];
ry(-3.0835180509818776) q[7];
cx q[6],q[7];
ry(-2.3602019108762486) q[8];
ry(1.7257628953729414) q[9];
cx q[8],q[9];
ry(1.4400649322758974) q[8];
ry(3.137658353345242) q[9];
cx q[8],q[9];
ry(-3.133062502619297) q[10];
ry(1.578858470745791) q[11];
cx q[10],q[11];
ry(1.570684417807391) q[10];
ry(1.579798449026109) q[11];
cx q[10],q[11];
ry(0.1436445011877404) q[0];
ry(1.6366735059384794) q[2];
cx q[0],q[2];
ry(2.9549572931701493) q[0];
ry(1.9023893699021468) q[2];
cx q[0],q[2];
ry(0.9072286449498659) q[2];
ry(-2.6040297016010627) q[4];
cx q[2],q[4];
ry(-0.9791765236416974) q[2];
ry(2.8069584939805163) q[4];
cx q[2],q[4];
ry(1.5091038048357328) q[4];
ry(-0.004462162431007335) q[6];
cx q[4],q[6];
ry(2.619132188229553) q[4];
ry(-0.005437660830732938) q[6];
cx q[4],q[6];
ry(-1.2637247198854276) q[6];
ry(-0.7919916551750892) q[8];
cx q[6],q[8];
ry(1.5882187110399923) q[6];
ry(3.1322744304849737) q[8];
cx q[6],q[8];
ry(0.005920738458433569) q[8];
ry(0.03393614197677603) q[10];
cx q[8],q[10];
ry(0.1556158047248706) q[8];
ry(-0.3296012030184876) q[10];
cx q[8],q[10];
ry(0.9409758944559039) q[1];
ry(1.772804440643072) q[3];
cx q[1],q[3];
ry(1.052966445397856) q[1];
ry(-1.0576978001736432) q[3];
cx q[1],q[3];
ry(2.4151437208485045) q[3];
ry(0.744573053572914) q[5];
cx q[3],q[5];
ry(-0.26002971376203204) q[3];
ry(1.122176699988449) q[5];
cx q[3],q[5];
ry(-2.6533721792669858) q[5];
ry(-2.526650997558797) q[7];
cx q[5],q[7];
ry(0.0021121538133597184) q[5];
ry(3.10787708490162) q[7];
cx q[5],q[7];
ry(-2.1594356762729054) q[7];
ry(1.5693175071776302) q[9];
cx q[7],q[9];
ry(-1.5590983847617288) q[7];
ry(1.5728899908712677) q[9];
cx q[7],q[9];
ry(0.33835289829354015) q[9];
ry(0.09977992779391442) q[11];
cx q[9],q[11];
ry(0.00040952735930899653) q[9];
ry(-3.1388867379975776) q[11];
cx q[9],q[11];
ry(-2.954711768901911) q[0];
ry(0.9297268949409784) q[3];
cx q[0],q[3];
ry(-1.799258711348167) q[0];
ry(1.820968588315366) q[3];
cx q[0],q[3];
ry(2.07590908734979) q[1];
ry(-1.466566968675437) q[2];
cx q[1],q[2];
ry(1.9939605335493802) q[1];
ry(-1.0794493376284566) q[2];
cx q[1],q[2];
ry(-2.1639838833099434) q[2];
ry(2.408609435167237) q[5];
cx q[2],q[5];
ry(-2.796132680637561) q[2];
ry(1.9189317537646027) q[5];
cx q[2],q[5];
ry(2.007300292287236) q[3];
ry(-2.247733264024288) q[4];
cx q[3],q[4];
ry(-0.011380901106594033) q[3];
ry(-1.0860483417464675) q[4];
cx q[3],q[4];
ry(-0.9967261325996235) q[4];
ry(-0.03755846377523486) q[7];
cx q[4],q[7];
ry(3.097875234107787) q[4];
ry(-3.1414396433862555) q[7];
cx q[4],q[7];
ry(2.8071698201432587) q[5];
ry(2.188874277781374) q[6];
cx q[5],q[6];
ry(3.1401111942575914) q[5];
ry(0.0042326838877970105) q[6];
cx q[5],q[6];
ry(1.7878396489469788) q[6];
ry(0.35399572793377576) q[9];
cx q[6],q[9];
ry(-2.995692740755944) q[6];
ry(3.1397654457361153) q[9];
cx q[6],q[9];
ry(-0.038526372358702154) q[7];
ry(-1.5750833749843385) q[8];
cx q[7],q[8];
ry(-1.5780062443521612) q[7];
ry(1.5559212951827102) q[8];
cx q[7],q[8];
ry(-0.011779438411352317) q[8];
ry(2.821504334386354) q[11];
cx q[8],q[11];
ry(3.1403509439832638) q[8];
ry(-0.018922659808565) q[11];
cx q[8],q[11];
ry(3.1287411653252164) q[9];
ry(1.5409623068351141) q[10];
cx q[9],q[10];
ry(1.5730813371776833) q[9];
ry(0.00019193111562749735) q[10];
cx q[9],q[10];
ry(0.2365406889387216) q[0];
ry(0.7796495519480795) q[1];
cx q[0],q[1];
ry(-1.7248877632607096) q[0];
ry(1.74964407871899) q[1];
cx q[0],q[1];
ry(-1.0412569883508054) q[2];
ry(-0.1557997080171275) q[3];
cx q[2],q[3];
ry(-1.553724335095975) q[2];
ry(-1.9137780052868645) q[3];
cx q[2],q[3];
ry(0.39564028326183204) q[4];
ry(-0.4959382683238728) q[5];
cx q[4],q[5];
ry(-1.77678272497816) q[4];
ry(-0.0021472912151949686) q[5];
cx q[4],q[5];
ry(-0.2886607599220201) q[6];
ry(1.574610458546999) q[7];
cx q[6],q[7];
ry(2.5214375246312137) q[6];
ry(-3.1388930364540677) q[7];
cx q[6],q[7];
ry(0.012797474329252267) q[8];
ry(0.029499015837828328) q[9];
cx q[8],q[9];
ry(-1.5877280136558778) q[8];
ry(-1.5506118067500485) q[9];
cx q[8],q[9];
ry(1.5676176094091157) q[10];
ry(-0.37626161012612425) q[11];
cx q[10],q[11];
ry(3.136054162632898) q[10];
ry(1.5628629305359283) q[11];
cx q[10],q[11];
ry(-2.765609452790805) q[0];
ry(-2.4218047127763436) q[2];
cx q[0],q[2];
ry(-0.6654053382345021) q[0];
ry(-1.3660592517853292) q[2];
cx q[0],q[2];
ry(-2.6011244965882514) q[2];
ry(0.7235357751881377) q[4];
cx q[2],q[4];
ry(3.140300903278281) q[2];
ry(-1.9220598668914681) q[4];
cx q[2],q[4];
ry(-3.137470478674652) q[4];
ry(2.956460846650336) q[6];
cx q[4],q[6];
ry(-0.780287920552786) q[4];
ry(1.5693969453560461) q[6];
cx q[4],q[6];
ry(2.6328419699176533) q[6];
ry(-3.099281286170381) q[8];
cx q[6],q[8];
ry(3.1395214462342835) q[6];
ry(7.732488678410115e-05) q[8];
cx q[6],q[8];
ry(1.3781126142437712) q[8];
ry(-0.39206852198442377) q[10];
cx q[8],q[10];
ry(-1.5807652688482454) q[8];
ry(-3.1392107191806016) q[10];
cx q[8],q[10];
ry(-1.5783184263996626) q[1];
ry(1.4274259469605726) q[3];
cx q[1],q[3];
ry(2.640197428444021) q[1];
ry(0.8035766928518271) q[3];
cx q[1],q[3];
ry(1.3694832411803954) q[3];
ry(-1.489302264116148) q[5];
cx q[3],q[5];
ry(1.3848597225791917) q[3];
ry(1.6654242487925313) q[5];
cx q[3],q[5];
ry(-1.1477189150683733) q[5];
ry(3.1279569921415553) q[7];
cx q[5],q[7];
ry(-3.1406906217280697) q[5];
ry(3.1413247335541468) q[7];
cx q[5],q[7];
ry(0.6254531831907189) q[7];
ry(0.022849641292552524) q[9];
cx q[7],q[9];
ry(-1.5719551739994548) q[7];
ry(-3.135550326745175) q[9];
cx q[7],q[9];
ry(-1.3210482441632678) q[9];
ry(-1.727513043737014) q[11];
cx q[9],q[11];
ry(1.5975068622807254) q[9];
ry(3.1404752626323593) q[11];
cx q[9],q[11];
ry(0.4845605637135959) q[0];
ry(-2.644076126074985) q[3];
cx q[0],q[3];
ry(1.425491464895129) q[0];
ry(2.9754560192811312) q[3];
cx q[0],q[3];
ry(-0.26369157948338007) q[1];
ry(1.1352852767891664) q[2];
cx q[1],q[2];
ry(-1.7327259435077302) q[1];
ry(2.057107001343632) q[2];
cx q[1],q[2];
ry(0.11046463324752233) q[2];
ry(2.0103873910994) q[5];
cx q[2],q[5];
ry(-3.0200277804318905) q[2];
ry(-2.4046058487425155) q[5];
cx q[2],q[5];
ry(1.129617629918685) q[3];
ry(1.5726347378264287) q[4];
cx q[3],q[4];
ry(1.5731075706571367) q[3];
ry(-1.5609305876239907) q[4];
cx q[3],q[4];
ry(2.6196978789525307) q[4];
ry(0.5735140669004258) q[7];
cx q[4],q[7];
ry(3.14042093970019) q[4];
ry(-3.140690291954785) q[7];
cx q[4],q[7];
ry(-2.693401878134606) q[5];
ry(2.0704297567076653) q[6];
cx q[5],q[6];
ry(1.5732567987946853) q[5];
ry(1.5667494152979005) q[6];
cx q[5],q[6];
ry(0.37517210805865336) q[6];
ry(-1.348755296308548) q[9];
cx q[6],q[9];
ry(0.004562518739765764) q[6];
ry(0.0001940993777007096) q[9];
cx q[6],q[9];
ry(3.0658810609502165) q[7];
ry(2.9070028722070766) q[8];
cx q[7],q[8];
ry(-1.551707696307461) q[7];
ry(-1.5300852507814284) q[8];
cx q[7],q[8];
ry(-0.0021579214052023744) q[8];
ry(-1.2148738694928702) q[11];
cx q[8],q[11];
ry(0.0032285090657460202) q[8];
ry(-1.568762407659115) q[11];
cx q[8],q[11];
ry(-3.1045640450775815) q[9];
ry(3.0882127676818794) q[10];
cx q[9],q[10];
ry(1.5696777420003374) q[9];
ry(-1.5739581204770798) q[10];
cx q[9],q[10];
ry(0.20387469738805405) q[0];
ry(1.1278162596474754) q[1];
ry(-2.2673017879014523) q[2];
ry(1.5682211630807323) q[3];
ry(-0.5242517207415398) q[4];
ry(3.1407815547597098) q[5];
ry(-0.3748136293002489) q[6];
ry(1.5774754721101272) q[7];
ry(3.1370499534668967) q[8];
ry(-3.1400569048692066) q[9];
ry(-1.7322559607337642) q[10];
ry(-0.3589196798404169) q[11];