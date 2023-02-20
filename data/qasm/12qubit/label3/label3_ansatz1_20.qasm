OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.0926521105847606) q[0];
rz(-1.681050323674691) q[0];
ry(-0.6011932351609545) q[1];
rz(2.9902738632395756) q[1];
ry(-2.786000190355135) q[2];
rz(-0.024665218269176097) q[2];
ry(0.001693697334964004) q[3];
rz(0.37252519706563725) q[3];
ry(0.6428632904675817) q[4];
rz(2.5851507880718003) q[4];
ry(-1.5192847686399782) q[5];
rz(-2.4359933870657566) q[5];
ry(2.5370150265603506) q[6];
rz(2.606018123040011) q[6];
ry(0.18718584056079735) q[7];
rz(2.1740361375205577) q[7];
ry(0.22348882830344607) q[8];
rz(1.9143417953819641) q[8];
ry(-0.36962252573325144) q[9];
rz(-3.037992046266362) q[9];
ry(1.4084719161379413) q[10];
rz(1.1564813850640483) q[10];
ry(-0.7341252623745995) q[11];
rz(0.3394811550857186) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.9728078753765086) q[0];
rz(-0.48468515615811203) q[0];
ry(-2.799526417621132) q[1];
rz(-0.3320101163936261) q[1];
ry(2.4501771795413316) q[2];
rz(-1.6936420625822226) q[2];
ry(-0.2750445026468783) q[3];
rz(1.762974736223339) q[3];
ry(-0.0023875470370002816) q[4];
rz(0.9391731199980222) q[4];
ry(0.008659477338254219) q[5];
rz(2.364713633616449) q[5];
ry(1.3387414316569544) q[6];
rz(-3.1122722206322533) q[6];
ry(0.235450515710365) q[7];
rz(-2.902852497948478) q[7];
ry(-1.4935783783212615) q[8];
rz(-3.0424453198319914) q[8];
ry(-0.03562780700200996) q[9];
rz(1.610846311292636) q[9];
ry(2.981888022727679) q[10];
rz(-0.6740970137447071) q[10];
ry(1.2642479523071561) q[11];
rz(2.868999282486078) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3934125265314936) q[0];
rz(-3.0146007207945753) q[0];
ry(1.179632650041842) q[1];
rz(-0.276314363608642) q[1];
ry(1.6258849514334217) q[2];
rz(1.9854582300088528) q[2];
ry(1.5332157967301345) q[3];
rz(1.4587243606349742) q[3];
ry(0.13368994981134286) q[4];
rz(1.7334474715661976) q[4];
ry(1.8216867568907682) q[5];
rz(-2.7104588637980433) q[5];
ry(1.7121421761723035) q[6];
rz(0.034569930879819216) q[6];
ry(-1.3963581528350821) q[7];
rz(-3.0354472161626522) q[7];
ry(-1.9018213467837137) q[8];
rz(-1.1593904855310808) q[8];
ry(-1.8236982573540876) q[9];
rz(2.5398119809897874) q[9];
ry(0.7924715535036331) q[10];
rz(-0.6491804912985293) q[10];
ry(-2.674874127476476) q[11];
rz(-1.6354500341997513) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.446473414886991) q[0];
rz(0.8079665906296718) q[0];
ry(-0.7157323451267541) q[1];
rz(0.2502543854191667) q[1];
ry(3.1017401267731053) q[2];
rz(-2.4473498758755974) q[2];
ry(0.027085271862869922) q[3];
rz(-0.7999374220786998) q[3];
ry(2.9074748549179046) q[4];
rz(-1.0655905861444266) q[4];
ry(-0.019133208336783013) q[5];
rz(-1.9553683811648475) q[5];
ry(0.5551256515814238) q[6];
rz(0.16217585778273505) q[6];
ry(0.043269612407867974) q[7];
rz(-1.5451076515783448) q[7];
ry(3.1368762645958377) q[8];
rz(1.357505069176173) q[8];
ry(3.0566402168892326) q[9];
rz(1.677996973284456) q[9];
ry(1.8271716444422088) q[10];
rz(-3.1123870788068753) q[10];
ry(0.373864461444195) q[11];
rz(3.1041822600031126) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.033108768411608) q[0];
rz(-1.6086729013171412) q[0];
ry(-3.129962507669719) q[1];
rz(2.992141515809025) q[1];
ry(0.1502686868757177) q[2];
rz(-0.03654573152959096) q[2];
ry(3.1396665343743) q[3];
rz(0.6548704427073719) q[3];
ry(1.1317968324530097) q[4];
rz(0.052443479111704334) q[4];
ry(3.06079506683288) q[5];
rz(1.5093724361718972) q[5];
ry(-2.991422688290534) q[6];
rz(0.2144834651848342) q[6];
ry(1.7650901825217444) q[7];
rz(-2.87260534264679) q[7];
ry(-2.7471433252867072) q[8];
rz(1.0842455682135759) q[8];
ry(0.03610630204176219) q[9];
rz(-1.5827611160818194) q[9];
ry(0.7938441923755608) q[10];
rz(-1.6799327776171067) q[10];
ry(2.015665679592859) q[11];
rz(-1.2230685936074455) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.5089047111044724) q[0];
rz(2.2694008588366663) q[0];
ry(-0.8216497628912718) q[1];
rz(-2.6691987342694476) q[1];
ry(-1.3799211336377286) q[2];
rz(2.906019644268638) q[2];
ry(-0.059420109138556346) q[3];
rz(-1.918675671144601) q[3];
ry(1.6832412089637707) q[4];
rz(-0.018161514940530525) q[4];
ry(0.03741626790748942) q[5];
rz(-3.072812928790117) q[5];
ry(-1.6999181221920623) q[6];
rz(3.055347230724161) q[6];
ry(1.3356360161997973) q[7];
rz(-0.0036930288772154536) q[7];
ry(-0.033211313227991646) q[8];
rz(-0.10225558182128047) q[8];
ry(3.111815941665089) q[9];
rz(2.52468755154944) q[9];
ry(-1.2059897623669693) q[10];
rz(1.2460100169588093) q[10];
ry(-0.09730299890733729) q[11];
rz(-0.6215301858051081) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.5800931187660918) q[0];
rz(-0.12829447059783033) q[0];
ry(-2.9932404435789435) q[1];
rz(2.1778250150747382) q[1];
ry(2.0693314524990165) q[2];
rz(1.4027471642657012) q[2];
ry(-3.113058174596826) q[3];
rz(-2.1852549585070964) q[3];
ry(-1.6600954996453432) q[4];
rz(2.7476599301121807) q[4];
ry(1.5713538907674476) q[5];
rz(-0.09524661813297744) q[5];
ry(-0.0022994367404312044) q[6];
rz(-1.4371902225599547) q[6];
ry(2.12409352127171) q[7];
rz(0.0472076652148032) q[7];
ry(-1.9069636083840216) q[8];
rz(2.7455952998028907) q[8];
ry(1.6838384947074634) q[9];
rz(0.16545413570225886) q[9];
ry(-2.944315754411695) q[10];
rz(0.47291089050076085) q[10];
ry(-1.0173889048914966) q[11];
rz(2.619677089234576) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.8820531639011837) q[0];
rz(-0.18766870709503153) q[0];
ry(0.3587161943236472) q[1];
rz(-1.7693198822631515) q[1];
ry(-0.7701693080883077) q[2];
rz(-0.8613602180976665) q[2];
ry(-0.7046987562031353) q[3];
rz(-1.4561607110925308) q[3];
ry(1.3542757049122256) q[4];
rz(1.7919476587635772) q[4];
ry(-1.7004749512587256) q[5];
rz(-0.0843808552702141) q[5];
ry(1.5964016182716874) q[6];
rz(3.1046856913205936) q[6];
ry(-0.40456418820211126) q[7];
rz(3.094874040458663) q[7];
ry(-0.0012207141576388025) q[8];
rz(0.3951714931023469) q[8];
ry(0.09714315934373308) q[9];
rz(1.5595992392828828) q[9];
ry(1.2444864034379854) q[10];
rz(-1.4843145854456345) q[10];
ry(3.066451794121665) q[11];
rz(3.0012536859419563) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.03751905404108058) q[0];
rz(-1.415969504887606) q[0];
ry(1.1918746873930366) q[1];
rz(0.4611793073968967) q[1];
ry(-0.4170763085951345) q[2];
rz(2.5843300119512183) q[2];
ry(-3.1327079703057352) q[3];
rz(1.865215403562436) q[3];
ry(-3.026550430078969) q[4];
rz(2.704040188094702) q[4];
ry(0.8843867534868491) q[5];
rz(1.9170524924940269) q[5];
ry(-0.08762037863405502) q[6];
rz(-0.8551162569074436) q[6];
ry(1.5689246290920598) q[7];
rz(0.5357696462826027) q[7];
ry(1.194197091362222) q[8];
rz(1.5670639053250837) q[8];
ry(1.5624473734296722) q[9];
rz(-2.6161806247966566) q[9];
ry(-2.872689273860949) q[10];
rz(2.1001722392644675) q[10];
ry(-1.6747465629420413) q[11];
rz(0.09459206269432574) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.939037925740198) q[0];
rz(1.885464668845489) q[0];
ry(2.8035603992869538) q[1];
rz(-2.0779578118707738) q[1];
ry(1.7357964283493381) q[2];
rz(1.9932434165194655) q[2];
ry(1.332719894721132) q[3];
rz(0.5095313664970877) q[3];
ry(3.0230163794516822) q[4];
rz(2.6101901067915785) q[4];
ry(-2.9335611743156766) q[5];
rz(2.000227441912146) q[5];
ry(-3.104406143746204) q[6];
rz(1.848999932472479) q[6];
ry(0.06098773195441165) q[7];
rz(-2.113873189125265) q[7];
ry(1.5699691078401123) q[8];
rz(2.606132273554267) q[8];
ry(1.5703363607896228) q[9];
rz(1.5781381642926595) q[9];
ry(-3.059060320721146) q[10];
rz(-2.7194296174665404) q[10];
ry(0.18902687699984186) q[11];
rz(1.7044955888199211) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1297396712028993) q[0];
rz(-2.4260756514702373) q[0];
ry(1.6287259201663584) q[1];
rz(-0.5075773940826807) q[1];
ry(-2.4298566262104937) q[2];
rz(-1.7277774954659586) q[2];
ry(0.002611383694341107) q[3];
rz(-0.21481292405490138) q[3];
ry(-2.750942552822529) q[4];
rz(-0.8898304168268455) q[4];
ry(-0.934222679372863) q[5];
rz(-1.7430644204800856) q[5];
ry(-0.005294006175129706) q[6];
rz(-0.9871494114077181) q[6];
ry(1.5496928960608978) q[7];
rz(0.435282619802881) q[7];
ry(0.012746469941001592) q[8];
rz(0.5171595866535108) q[8];
ry(1.5703048410075449) q[9];
rz(3.107009578925966) q[9];
ry(2.404824850202556) q[10];
rz(3.1202817610647475) q[10];
ry(-2.029860921916318) q[11];
rz(2.8245737104190596) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.436938957211721) q[0];
rz(-0.8122205388283188) q[0];
ry(1.2650371367734394) q[1];
rz(-2.8019219992861144) q[1];
ry(-1.9728723660769858) q[2];
rz(1.184994557761162) q[2];
ry(1.873902118409994) q[3];
rz(-2.421352915675817) q[3];
ry(-1.851116001981653) q[4];
rz(-1.8546812431490682) q[4];
ry(1.6059266724464836) q[5];
rz(2.3196610348074835) q[5];
ry(1.6136524729128527) q[6];
rz(2.9191278952220987) q[6];
ry(3.110761203793295) q[7];
rz(-2.568883894363918) q[7];
ry(-1.5622908017040285) q[8];
rz(2.2933649674451253) q[8];
ry(-0.41370022566811565) q[9];
rz(-1.5227368666326582) q[9];
ry(1.5494358830530253) q[10];
rz(0.021570382340894593) q[10];
ry(0.09629171255376126) q[11];
rz(2.653530600563667) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.000990819249284236) q[0];
rz(2.135659691386663) q[0];
ry(-2.937085881902851) q[1];
rz(-1.3103384477652957) q[1];
ry(-0.11738457051555429) q[2];
rz(-1.9837584691180166) q[2];
ry(-0.013931320322172702) q[3];
rz(0.8866083645230505) q[3];
ry(2.8994681207100337) q[4];
rz(-3.087439396424904) q[4];
ry(-3.13230192618411) q[5];
rz(0.6991809640516927) q[5];
ry(-3.1177491195793556) q[6];
rz(1.3478668938433929) q[6];
ry(2.879367804452711) q[7];
rz(-3.091462572715965) q[7];
ry(0.33347222898482887) q[8];
rz(-0.9755300749915934) q[8];
ry(0.05621820136888503) q[9];
rz(3.1201552469044556) q[9];
ry(-2.28786398569558) q[10];
rz(0.08773224947946723) q[10];
ry(0.1462451802705497) q[11];
rz(-1.6780774835504793) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.8858592316350162) q[0];
rz(1.5534898575429539) q[0];
ry(2.31827305716423) q[1];
rz(3.0289821629472646) q[1];
ry(2.6756548133848455) q[2];
rz(1.1861900239684748) q[2];
ry(-1.6038124518098626) q[3];
rz(1.617022488291325) q[3];
ry(-0.758418713625252) q[4];
rz(1.5026716131438191) q[4];
ry(-1.4050218336857745) q[5];
rz(-0.03387585611790783) q[5];
ry(-1.6271550943058195) q[6];
rz(-0.03952631359712163) q[6];
ry(1.5976445019675873) q[7];
rz(-0.9617793573675303) q[7];
ry(-1.47164139997858) q[8];
rz(-2.8867843790818037) q[8];
ry(-1.5450017687488364) q[9];
rz(1.7706510721061586) q[9];
ry(0.02176670306197944) q[10];
rz(1.9548636752755968) q[10];
ry(-0.07016791599370852) q[11];
rz(2.5876034535994394) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.0012940405004115976) q[0];
rz(-3.1385270356001267) q[0];
ry(-3.07335475097668) q[1];
rz(3.0835484577587513) q[1];
ry(-0.3565755138415146) q[2];
rz(1.4248490909672087) q[2];
ry(-1.5760687114541385) q[3];
rz(-3.1178139708576533) q[3];
ry(-0.38453331005937974) q[4];
rz(1.5906179634667001) q[4];
ry(1.571022696933837) q[5];
rz(-0.9801208825311741) q[5];
ry(-3.0937521925141733) q[6];
rz(-0.43002412369449416) q[6];
ry(-3.1269789800845773) q[7];
rz(-0.9392328054033562) q[7];
ry(-0.48339202109487545) q[8];
rz(1.4762594105863498) q[8];
ry(0.007067350981101229) q[9];
rz(-2.5044722496574945) q[9];
ry(1.7161423346657965) q[10];
rz(1.763052127107021) q[10];
ry(-1.660384767723654) q[11];
rz(-1.0166505300040196) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.30349467692810433) q[0];
rz(1.7657549656472198) q[0];
ry(1.6707493307142658) q[1];
rz(-2.9978235913825593) q[1];
ry(1.5607727854435662) q[2];
rz(0.05493343145865826) q[2];
ry(-1.851001463500503) q[3];
rz(-3.135944260770525) q[3];
ry(-1.4434769136285253) q[4];
rz(3.1350444633927665) q[4];
ry(-3.1145421234698434) q[5];
rz(-0.897377244059452) q[5];
ry(0.1087839670296944) q[6];
rz(-2.7400924396668276) q[6];
ry(0.16819325076015623) q[7];
rz(0.1937025457784758) q[7];
ry(0.630288358162911) q[8];
rz(1.7088102112788521) q[8];
ry(3.139140665099142) q[9];
rz(0.6909769014644928) q[9];
ry(-2.096668432357417) q[10];
rz(-2.43259069870783) q[10];
ry(-1.1585394523214632) q[11];
rz(1.8393364546154256) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1404183442785523) q[0];
rz(0.09665511528332171) q[0];
ry(1.3159725523710568) q[1];
rz(-0.36563302950179905) q[1];
ry(-2.8550856991496616) q[2];
rz(-1.121859538014933) q[2];
ry(1.568026396201109) q[3];
rz(1.529150139697344) q[3];
ry(-3.1090695851392867) q[4];
rz(3.1358444085054624) q[4];
ry(-0.0014814223336703236) q[5];
rz(1.4727473425966453) q[5];
ry(3.090075177088693) q[6];
rz(-2.8965255098465743) q[6];
ry(3.11203085448702) q[7];
rz(-0.08706865542750906) q[7];
ry(0.3987366849713494) q[8];
rz(2.497347493788249) q[8];
ry(-1.4797380638850046) q[9];
rz(1.658910624063339) q[9];
ry(0.03838925063454646) q[10];
rz(-1.6967228086985422) q[10];
ry(0.003669629598752166) q[11];
rz(-1.9422896359687583) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.2546462724938445) q[0];
rz(1.2399170409973046) q[0];
ry(1.8066209673728988) q[1];
rz(3.0158868651859843) q[1];
ry(-0.00031183294143636344) q[2];
rz(2.731979483939919) q[2];
ry(1.560677419741138) q[3];
rz(1.3181114741718423) q[3];
ry(1.5847453837469978) q[4];
rz(1.7402349028837536) q[4];
ry(-1.526300622406468) q[5];
rz(0.04103082094192309) q[5];
ry(-0.20569122395692677) q[6];
rz(-0.24793999763605304) q[6];
ry(-1.3351953454955154) q[7];
rz(-1.8378327117606914) q[7];
ry(0.7326824617136795) q[8];
rz(-2.293727557729919) q[8];
ry(-0.018531499446613253) q[9];
rz(2.21927055504805) q[9];
ry(1.4582937356225918) q[10];
rz(-1.5197136073720081) q[10];
ry(1.1429268565863169) q[11];
rz(-1.693851188346951) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1409898322449465) q[0];
rz(2.8707038713519424) q[0];
ry(-2.8051525041437952) q[1];
rz(2.2663215331715234) q[1];
ry(-3.1205999367637864) q[2];
rz(3.1017612955927896) q[2];
ry(-1.5566400159184823) q[3];
rz(0.3696023467197991) q[3];
ry(-0.09639174806194005) q[4];
rz(0.0020625030785121692) q[4];
ry(3.1365258567354153) q[5];
rz(1.3492062236806095) q[5];
ry(3.12498195389793) q[6];
rz(0.9201968251198007) q[6];
ry(3.116725291098843) q[7];
rz(-3.0700673447260445) q[7];
ry(-0.08119517372123602) q[8];
rz(2.339583018521669) q[8];
ry(0.11002487228133084) q[9];
rz(-1.9452428194302875) q[9];
ry(1.5513737322997445) q[10];
rz(2.791554088160053) q[10];
ry(-0.018873956417361597) q[11];
rz(-0.1995848762427783) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5264107915105587) q[0];
rz(-0.35940350462283965) q[0];
ry(-0.23958465491468398) q[1];
rz(-0.7753240585310058) q[1];
ry(-1.5862818120094155) q[2];
rz(-1.2872203581648982) q[2];
ry(-0.071048246943902) q[3];
rz(-2.3006980519514095) q[3];
ry(1.6625190069751694) q[4];
rz(-3.1380466416601323) q[4];
ry(-3.0207356311470313) q[5];
rz(0.9194569252360719) q[5];
ry(0.8257505779086614) q[6];
rz(1.2332165245277917) q[6];
ry(2.0422491986335407) q[7];
rz(-2.4521932780392426) q[7];
ry(2.4163494079540078) q[8];
rz(-1.6276756901745042) q[8];
ry(-2.308395466931706) q[9];
rz(-0.5358172329410164) q[9];
ry(-0.18378735383568698) q[10];
rz(-0.5905615200763741) q[10];
ry(2.692083233042887) q[11];
rz(-0.3930583934180695) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1411105632579632) q[0];
rz(2.0943022403743026) q[0];
ry(-1.5525421762713483) q[1];
rz(0.03779662353084119) q[1];
ry(-3.061488454845546) q[2];
rz(-1.0709443987527651) q[2];
ry(0.0029874984002491974) q[3];
rz(-0.47029577107545484) q[3];
ry(-3.0462628373968554) q[4];
rz(-1.4521724105204736) q[4];
ry(3.1289313076242267) q[5];
rz(-0.06527892512745553) q[5];
ry(3.1378335475259895) q[6];
rz(-0.8971792126196011) q[6];
ry(3.1410625147657467) q[7];
rz(-1.833004020962968) q[7];
ry(2.942138814706918) q[8];
rz(1.5595677512548702) q[8];
ry(2.445548681163779) q[9];
rz(2.24430056132575) q[9];
ry(3.001414495322601) q[10];
rz(1.3314684148371012) q[10];
ry(-3.123585104064925) q[11];
rz(-2.7034469703474118) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.917286360528708) q[0];
rz(-0.8141291205198893) q[0];
ry(2.1884360723234826) q[1];
rz(0.3734577423530132) q[1];
ry(0.5993227268684249) q[2];
rz(-3.0989263085283962) q[2];
ry(-1.3492925067425192) q[3];
rz(-0.5511312445608105) q[3];
ry(-2.8272014952091804) q[4];
rz(2.314912237684541) q[4];
ry(-1.8462548470474247) q[5];
rz(-0.19342754902286097) q[5];
ry(-0.7403765649115819) q[6];
rz(-1.495482476409062) q[6];
ry(-0.2980022053891167) q[7];
rz(1.5188149853069053) q[7];
ry(0.17003577446078122) q[8];
rz(-1.4528094879236566) q[8];
ry(0.16795038875758106) q[9];
rz(-2.183053473242496) q[9];
ry(-0.010223265797457692) q[10];
rz(-0.13595757141301237) q[10];
ry(-1.0139732232857526) q[11];
rz(-1.198917399220943) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.00016023391080022265) q[0];
rz(-0.46517155911448993) q[0];
ry(0.02403633940707195) q[1];
rz(-0.5485191248541437) q[1];
ry(3.1132689825862125) q[2];
rz(-2.547014742507211) q[2];
ry(-7.810147901803927e-05) q[3];
rz(-1.0514587805446345) q[3];
ry(0.0006101608859797494) q[4];
rz(0.743839767539875) q[4];
ry(3.134131976145663) q[5];
rz(-1.2132581349790428) q[5];
ry(3.1348893342244093) q[6];
rz(-0.15901614380054951) q[6];
ry(0.014118759861429105) q[7];
rz(-2.670225690799854) q[7];
ry(0.17677106544491394) q[8];
rz(-0.034591486506721836) q[8];
ry(0.7074445068321558) q[9];
rz(-1.0382453394036535) q[9];
ry(-3.088429609259739) q[10];
rz(-3.087137095929256) q[10];
ry(-0.001933919278671138) q[11];
rz(-0.3785734436167827) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.305711760389942) q[0];
rz(-1.7104346441638227) q[0];
ry(-1.9284543071358207) q[1];
rz(-0.3954871673296006) q[1];
ry(-2.2853947363470475) q[2];
rz(3.1255197753906874) q[2];
ry(-2.875518988165751) q[3];
rz(-0.4287568738629336) q[3];
ry(-2.2661579425343845) q[4];
rz(1.2436939148420332) q[4];
ry(2.1182794444735116) q[5];
rz(0.09350376992655278) q[5];
ry(-1.0371223669675351) q[6];
rz(-2.449570841551022) q[6];
ry(2.0117685763849273) q[7];
rz(-1.4117955777570161) q[7];
ry(0.8301077907738096) q[8];
rz(1.914398400152691) q[8];
ry(1.4651377753467412) q[9];
rz(2.521002416309439) q[9];
ry(-1.024944576414705) q[10];
rz(2.3354586688415235) q[10];
ry(0.0993757108349147) q[11];
rz(2.4313584093125784) q[11];