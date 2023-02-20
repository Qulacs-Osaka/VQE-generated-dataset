OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.9990439218371051) q[0];
rz(0.720707642467219) q[0];
ry(1.6144611266359654) q[1];
rz(-0.26744622766409326) q[1];
ry(-2.9840380048579247) q[2];
rz(-0.7366166609703474) q[2];
ry(-0.11576229176347219) q[3];
rz(0.8831342943529847) q[3];
ry(-1.252396441449318) q[4];
rz(1.4300873403380212) q[4];
ry(3.118965486707147) q[5];
rz(-2.0474421821316287) q[5];
ry(0.7391518331841892) q[6];
rz(3.1300705859985123) q[6];
ry(-0.18994377015207403) q[7];
rz(-1.0876947455821901) q[7];
ry(-2.811529094169499) q[8];
rz(-0.21400940699926618) q[8];
ry(-2.8839967959887516) q[9];
rz(1.4076052340619967) q[9];
ry(2.772327829274869) q[10];
rz(0.7660624636015189) q[10];
ry(0.6884755844385486) q[11];
rz(-0.2138400880105771) q[11];
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
ry(1.589986987309406) q[0];
rz(0.894452218127121) q[0];
ry(0.7298248836121664) q[1];
rz(-0.23895564497883393) q[1];
ry(-1.2972791299490067) q[2];
rz(-1.3807140344397122) q[2];
ry(1.4416033317279682) q[3];
rz(0.3828174933760744) q[3];
ry(-2.615854914888644) q[4];
rz(-0.5113439143743079) q[4];
ry(-0.8489587681139364) q[5];
rz(-0.3051201329608917) q[5];
ry(-0.6483054924893734) q[6];
rz(2.6218519943962133) q[6];
ry(0.23982652238400082) q[7];
rz(0.9270661720151212) q[7];
ry(-2.736991502176034) q[8];
rz(-2.8529456746006794) q[8];
ry(-1.3757935311981946) q[9];
rz(-0.1953717390646753) q[9];
ry(-0.08472780676113961) q[10];
rz(-1.799565920791827) q[10];
ry(0.6558729412445632) q[11];
rz(1.899605315798197) q[11];
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
ry(-0.3089603342211558) q[0];
rz(1.3710767046894372) q[0];
ry(1.6320015279860582) q[1];
rz(-2.169317700396192) q[1];
ry(-1.4833783867335166) q[2];
rz(-1.6458152279209894) q[2];
ry(-0.8590722084258732) q[3];
rz(-0.948900879740041) q[3];
ry(-0.6150100886731885) q[4];
rz(-1.3844971177271974) q[4];
ry(0.5501726776842251) q[5];
rz(3.1065304820705126) q[5];
ry(-3.138590623835006) q[6];
rz(-3.1256609265131114) q[6];
ry(0.5423739181358468) q[7];
rz(0.7336513310146904) q[7];
ry(3.115796596749895) q[8];
rz(-0.9461758110278488) q[8];
ry(1.8410036713056463) q[9];
rz(2.064863281906052) q[9];
ry(-0.8633641775971566) q[10];
rz(0.5921045590331071) q[10];
ry(-1.5013705180158563) q[11];
rz(1.5083657207502048) q[11];
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
ry(0.7790709321242765) q[0];
rz(1.356084201635623) q[0];
ry(1.621179766667363) q[1];
rz(-2.1918356088068696) q[1];
ry(-0.7152139642323423) q[2];
rz(0.35308863514423994) q[2];
ry(-2.7808465008459975) q[3];
rz(2.969782506717065) q[3];
ry(-3.05415701168958) q[4];
rz(2.538371515710879) q[4];
ry(-0.9555535783516095) q[5];
rz(-0.5898917817259135) q[5];
ry(-0.02221049393690285) q[6];
rz(-0.30160670525696076) q[6];
ry(0.9874791162304125) q[7];
rz(-3.1076160416772782) q[7];
ry(-1.4319068894096576) q[8];
rz(-1.551972483299103) q[8];
ry(-2.8364434721152207) q[9];
rz(-1.2936596230955288) q[9];
ry(0.9579094664755212) q[10];
rz(-0.9337661487662623) q[10];
ry(0.5588509475576438) q[11];
rz(-2.736875800589598) q[11];
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
ry(2.8195928879349883) q[0];
rz(-2.033464028127242) q[0];
ry(2.1742789843365076) q[1];
rz(-2.413212769738904) q[1];
ry(-1.2096091164161282) q[2];
rz(-1.0459213917552144) q[2];
ry(0.3320724628479912) q[3];
rz(3.0564108005084165) q[3];
ry(1.7414163154386573) q[4];
rz(1.616303620792944) q[4];
ry(2.1075709795112116) q[5];
rz(-2.1440091176595364) q[5];
ry(-3.065746271266026) q[6];
rz(1.6629739448326042) q[6];
ry(0.04045846039213963) q[7];
rz(-1.673103575363462) q[7];
ry(1.318681856691439) q[8];
rz(-2.118423403125872) q[8];
ry(0.5482121212283645) q[9];
rz(-2.1392276250114275) q[9];
ry(1.1286744934510329) q[10];
rz(-0.3763614110510298) q[10];
ry(1.0390307251049162) q[11];
rz(-2.246848433241169) q[11];
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
ry(1.6366009691656833) q[0];
rz(-1.1823608001187411) q[0];
ry(-0.34307853914883335) q[1];
rz(2.1655411678735677) q[1];
ry(-1.6747673772474707) q[2];
rz(-0.2544472126491284) q[2];
ry(2.1333773702441374) q[3];
rz(-1.5460365434039725) q[3];
ry(2.3725585483505442) q[4];
rz(-0.5632062803970843) q[4];
ry(3.0416419721956074) q[5];
rz(2.1393145171209422) q[5];
ry(2.627238876439498) q[6];
rz(3.108420787724323) q[6];
ry(-3.0866565056165536) q[7];
rz(-2.653359138077415) q[7];
ry(-0.9190608798177741) q[8];
rz(0.7096216961374745) q[8];
ry(-0.001700212500151288) q[9];
rz(2.9746869600143904) q[9];
ry(2.78800815450623) q[10];
rz(-0.7598370120200038) q[10];
ry(0.7935832708208785) q[11];
rz(1.0969782764960243) q[11];
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
ry(2.013599874876738) q[0];
rz(-0.36322463830821966) q[0];
ry(2.7002896525917692) q[1];
rz(0.26367504896130983) q[1];
ry(1.9414226021170904) q[2];
rz(-1.7020159608914098) q[2];
ry(2.7602749013977985) q[3];
rz(3.023241733194478) q[3];
ry(2.3369120896069013) q[4];
rz(1.6621199507490914) q[4];
ry(3.1296442919664815) q[5];
rz(-2.307380657709996) q[5];
ry(1.5632378391888109) q[6];
rz(-1.6567709839165135) q[6];
ry(-3.0906869961978107) q[7];
rz(2.7396025599729654) q[7];
ry(-1.4779525904281328) q[8];
rz(-1.0647908394573358) q[8];
ry(2.48193829578818) q[9];
rz(0.6688460526400863) q[9];
ry(1.3005585675604638) q[10];
rz(2.7464284917365203) q[10];
ry(-2.112229243913917) q[11];
rz(-1.9849053605542455) q[11];
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
ry(0.4218684087439968) q[0];
rz(2.880091719923943) q[0];
ry(1.5879173479893582) q[1];
rz(1.4948460320378039) q[1];
ry(1.4371930389127716) q[2];
rz(-2.5040581525788133) q[2];
ry(-1.8482022394678483) q[3];
rz(-1.6600531490317048) q[3];
ry(-1.1878232510438185) q[4];
rz(2.1753146288811434) q[4];
ry(-0.008426710552990713) q[5];
rz(2.342019476946536) q[5];
ry(-1.522631791927604) q[6];
rz(2.6041518184110863) q[6];
ry(3.1401319102125895) q[7];
rz(-0.3624508561932282) q[7];
ry(-1.6493605211898235) q[8];
rz(2.4293126045011393) q[8];
ry(-0.015793014942828698) q[9];
rz(-0.6955126061863918) q[9];
ry(3.1247357660943638) q[10];
rz(0.13696675824327006) q[10];
ry(-1.449862812958835) q[11];
rz(1.5349727247689104) q[11];
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
ry(-0.6466868477684645) q[0];
rz(1.3496914853970612) q[0];
ry(1.3060570910520892) q[1];
rz(0.6897754944391767) q[1];
ry(1.0994195365440782) q[2];
rz(2.069625343946681) q[2];
ry(0.30354888267979163) q[3];
rz(1.2881615329286031) q[3];
ry(-0.44690174816005285) q[4];
rz(-0.2907226384938886) q[4];
ry(-1.5043528472279917) q[5];
rz(-3.1412623719139865) q[5];
ry(0.6446627535414539) q[6];
rz(0.9873802046607316) q[6];
ry(-1.5708043554299331) q[7];
rz(1.6355475420609915) q[7];
ry(-1.9978630080142885) q[8];
rz(-3.069242493366918) q[8];
ry(2.4472053975285473) q[9];
rz(-2.0480324790411775) q[9];
ry(-1.0152307224314887) q[10];
rz(0.32465626158679406) q[10];
ry(-1.7851527322124436) q[11];
rz(-1.7983671062975592) q[11];
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
ry(1.0667069626289256) q[0];
rz(1.398609172724415) q[0];
ry(-0.46758425999090464) q[1];
rz(-2.245392306990044) q[1];
ry(-1.4916635096127944) q[2];
rz(-0.9112111874447653) q[2];
ry(1.374591634560576) q[3];
rz(2.437748914667498) q[3];
ry(-3.141290407908761) q[4];
rz(-1.8729412406120656) q[4];
ry(1.4922015917184348) q[5];
rz(-0.1179809842223654) q[5];
ry(3.141536013069633) q[6];
rz(-2.1966254285185567) q[6];
ry(-0.014343901010816397) q[7];
rz(-1.6430222730266855) q[7];
ry(1.5660213531936682) q[8];
rz(-3.1408110581961175) q[8];
ry(-1.5777282421879075) q[9];
rz(1.6162734778126011) q[9];
ry(-0.20395513627781625) q[10];
rz(-2.1192817413352363) q[10];
ry(-2.495148584887773) q[11];
rz(-0.43036326817481463) q[11];
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
ry(-0.43887272153619517) q[0];
rz(1.7407466623152457) q[0];
ry(-1.3434265750817156) q[1];
rz(-0.1381520374719063) q[1];
ry(1.384312194541751) q[2];
rz(0.24110289434724308) q[2];
ry(3.04309608564153) q[3];
rz(-2.5797787198786386) q[3];
ry(-3.0776112436556025) q[4];
rz(1.5763081653203743) q[4];
ry(0.0672093972866844) q[5];
rz(-3.0199608063751806) q[5];
ry(-2.3551825310134045) q[6];
rz(-3.1414872796331013) q[6];
ry(2.2600469182042406) q[7];
rz(1.4361604425611336) q[7];
ry(-1.6139404112721052) q[8];
rz(-1.5712021242954657) q[8];
ry(-1.571377228620996) q[9];
rz(-1.5719582346438763) q[9];
ry(1.5437775740022703) q[10];
rz(2.065356863662415) q[10];
ry(0.6114238434342756) q[11];
rz(2.659252258078418) q[11];
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
ry(-2.1961664825972553) q[0];
rz(-1.8345016515610173) q[0];
ry(-1.7405556799576907) q[1];
rz(0.23035564118857202) q[1];
ry(-1.9649417703838126) q[2];
rz(0.8724052136618377) q[2];
ry(-1.6576593731159874) q[3];
rz(-0.9282236810460225) q[3];
ry(-3.0319677918430123) q[4];
rz(-3.123442027396661) q[4];
ry(-1.6446008268931411) q[5];
rz(-1.3203085599426814) q[5];
ry(-0.8955558887190145) q[6];
rz(-0.11210096199383646) q[6];
ry(-3.139252268724608) q[7];
rz(-0.13131262738438265) q[7];
ry(1.5692564932433954) q[8];
rz(1.5588925701176717) q[8];
ry(-1.5674532884453045) q[9];
rz(-1.3956462211435967) q[9];
ry(0.0006799041886428239) q[10];
rz(-1.972660122180922) q[10];
ry(3.1381390370156823) q[11];
rz(-1.1454767084634243) q[11];
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
ry(1.8781671814411371) q[0];
rz(0.5162351803470857) q[0];
ry(-2.6215215249226387) q[1];
rz(-1.9380184629697685) q[1];
ry(-0.45426614777298796) q[2];
rz(0.4146618561420015) q[2];
ry(-3.0997485965019904) q[3];
rz(0.9004508931642502) q[3];
ry(2.964372482342072) q[4];
rz(2.841317198429008) q[4];
ry(3.1409375869857215) q[5];
rz(1.7399685098479685) q[5];
ry(0.02127865106625329) q[6];
rz(0.16145581845640233) q[6];
ry(-1.13285243284002) q[7];
rz(3.1379983795674065) q[7];
ry(1.5909323025140405) q[8];
rz(-1.6169273333180154) q[8];
ry(-2.2151169208601185) q[9];
rz(0.04438376349424498) q[9];
ry(-1.5722607944343103) q[10];
rz(-0.55767707564429) q[10];
ry(-0.46722823918308176) q[11];
rz(2.2106367931150674) q[11];
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
ry(1.1375942301787034) q[0];
rz(0.8538598175889338) q[0];
ry(-2.65990684390655) q[1];
rz(0.26184057252782283) q[1];
ry(0.7387527426565521) q[2];
rz(1.5952356595632784) q[2];
ry(-1.0272087166393509) q[3];
rz(1.1615763056291277) q[3];
ry(-1.2137229152830198) q[4];
rz(-1.1160704183376158) q[4];
ry(0.6725020148039755) q[5];
rz(0.25991467366486226) q[5];
ry(0.5490002511996249) q[6];
rz(-0.9812034623612663) q[6];
ry(-1.6283285438442157) q[7];
rz(0.10693610035033331) q[7];
ry(0.00037339434178038235) q[8];
rz(3.0997808070941604) q[8];
ry(1.5228880710929804) q[9];
rz(-2.4655104046138065) q[9];
ry(0.00025728598672536407) q[10];
rz(2.2259584205526126) q[10];
ry(-2.5610380060479034) q[11];
rz(-0.14466873994968477) q[11];
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
ry(-1.4079260641248372) q[0];
rz(0.8183287817134423) q[0];
ry(1.545366223132584) q[1];
rz(-1.4790825958636733) q[1];
ry(1.2260208853609231) q[2];
rz(1.125604554332039) q[2];
ry(-0.001014266692155985) q[3];
rz(0.9599621163055199) q[3];
ry(0.001901269249453108) q[4];
rz(-1.4007176844043432) q[4];
ry(-3.137661207932122) q[5];
rz(0.25360315047944143) q[5];
ry(-0.001583157548171954) q[6];
rz(-2.148530379852784) q[6];
ry(3.136752308986902) q[7];
rz(2.103489441112356) q[7];
ry(-1.6861509591377652) q[8];
rz(-2.019373786902928) q[8];
ry(2.3536034649872515) q[9];
rz(2.373812802035496) q[9];
ry(0.014722696675225022) q[10];
rz(-1.6172160368299737) q[10];
ry(-2.327166970915568) q[11];
rz(-0.5396955588866118) q[11];
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
ry(1.539613715097568) q[0];
rz(0.011042735153536043) q[0];
ry(-1.5831224047785888) q[1];
rz(1.6314332577515869) q[1];
ry(-1.9703845121871888) q[2];
rz(2.900404925862766) q[2];
ry(-3.106627310892664) q[3];
rz(2.2093858525945107) q[3];
ry(2.5829156982861616) q[4];
rz(-2.554153690484067) q[4];
ry(-0.6740290255313992) q[5];
rz(-2.6433630727845445) q[5];
ry(0.7870818152659236) q[6];
rz(0.6505299805320179) q[6];
ry(-0.021098587331956634) q[7];
rz(-2.251355728885777) q[7];
ry(-3.1408848233442033) q[8];
rz(-1.919544605430832) q[8];
ry(-2.941162124233529) q[9];
rz(-2.857025412312347) q[9];
ry(-0.12301790297064936) q[10];
rz(1.5373685942754929) q[10];
ry(-2.1126164101127136) q[11];
rz(1.4289512282152985) q[11];
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
ry(1.6461860300851106) q[0];
rz(-1.463613721421404) q[0];
ry(-2.327621841138819) q[1];
rz(1.6468016122715503) q[1];
ry(-0.027370722997193682) q[2];
rz(2.6423059204812) q[2];
ry(1.5736086609315842) q[3];
rz(-0.008905593598311107) q[3];
ry(1.5863829473476327) q[4];
rz(2.8512677525334444) q[4];
ry(0.0037470675766471118) q[5];
rz(2.839110407303214) q[5];
ry(-3.121968918282773) q[6];
rz(0.820458684694403) q[6];
ry(-0.0013519762893388076) q[7];
rz(0.01802056135285115) q[7];
ry(-0.1814522677263731) q[8];
rz(-0.10159187835176059) q[8];
ry(1.5567860380728638) q[9];
rz(-0.01769989655418217) q[9];
ry(1.7466680893107684) q[10];
rz(-1.6758832290437278) q[10];
ry(-1.7756603495732026) q[11];
rz(-2.7975570442634266) q[11];
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
ry(1.586106914036632) q[0];
rz(-2.9752730682174118) q[0];
ry(-0.5524171620542225) q[1];
rz(-1.6043187168859143) q[1];
ry(1.5748874307436158) q[2];
rz(-1.5692099817197416) q[2];
ry(-0.05894542652787216) q[3];
rz(-1.995441082954885) q[3];
ry(0.02280304042762271) q[4];
rz(1.5213453648111879) q[4];
ry(3.096073797149163) q[5];
rz(0.09438104490496729) q[5];
ry(-3.0884305174944435) q[6];
rz(2.686800320785006) q[6];
ry(-3.0587373295325135) q[7];
rz(1.4041189532074625) q[7];
ry(1.569834133561954) q[8];
rz(-3.139979416678076) q[8];
ry(-0.8111231667479268) q[9];
rz(0.00019901669102928565) q[9];
ry(0.002336695796071986) q[10];
rz(-1.4640602123197772) q[10];
ry(-1.5713120139918901) q[11];
rz(-2.9516983378889505) q[11];
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
ry(1.0176512034169134) q[0];
rz(0.7421550601878039) q[0];
ry(1.5707746591082004) q[1];
rz(1.5418895520618336) q[1];
ry(1.5709285510186854) q[2];
rz(2.8469905530427155) q[2];
ry(3.087205253746737) q[3];
rz(1.1166213767115825) q[3];
ry(-0.001930715676106587) q[4];
rz(-0.10426428032795668) q[4];
ry(3.137728134631995) q[5];
rz(-0.7385697936094269) q[5];
ry(3.1176551268001904) q[6];
rz(1.7643387980621643) q[6];
ry(-3.1410274069620185) q[7];
rz(0.06542386220401841) q[7];
ry(-1.5067008845436174) q[8];
rz(0.2649424034195872) q[8];
ry(1.5749828788884679) q[9];
rz(-1.6243674523489906) q[9];
ry(-1.568560014433298) q[10];
rz(-1.9819105967541328) q[10];
ry(-1.3065315247370257) q[11];
rz(1.7231823834957831) q[11];
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
ry(-0.17318007813477276) q[0];
rz(-0.32570780978938574) q[0];
ry(0.966471921208491) q[1];
rz(-1.692829784856388) q[1];
ry(-3.1405041206716446) q[2];
rz(1.1973949246700026) q[2];
ry(-1.5724250768348094) q[3];
rz(-0.00128988936801111) q[3];
ry(-3.1341846399850457) q[4];
rz(1.6022027924946458) q[4];
ry(2.481775816684172) q[5];
rz(-2.606554278849622) q[5];
ry(-1.7738617381917685) q[6];
rz(2.9901739585187497) q[6];
ry(-1.019406922623535) q[7];
rz(2.8339376019842994) q[7];
ry(-0.7162774720201206) q[8];
rz(-1.9628552181623897) q[8];
ry(-1.2078278956336508) q[9];
rz(-1.7137006960779304) q[9];
ry(-3.1400326571188915) q[10];
rz(1.2711787250911373) q[10];
ry(3.1242439439513574) q[11];
rz(1.79018231527382) q[11];
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
ry(-0.026360316861571354) q[0];
rz(0.32169688844004796) q[0];
ry(3.140964924456927) q[1];
rz(3.0014254494306947) q[1];
ry(3.091416690395359) q[2];
rz(-2.7729138582344732) q[2];
ry(1.1787718632492006) q[3];
rz(-1.5458979672802675) q[3];
ry(-3.1415368806073647) q[4];
rz(2.0442605482868457) q[4];
ry(-0.002707837669795168) q[5];
rz(-2.5589981952833667) q[5];
ry(-3.1405142090187343) q[6];
rz(-1.5298292001556755) q[6];
ry(3.1410292212542683) q[7];
rz(2.822543792763734) q[7];
ry(0.0010369087492358038) q[8];
rz(0.18915040498347005) q[8];
ry(-3.140495902767613) q[9];
rz(-1.863905178332514) q[9];
ry(-0.0010075525846735545) q[10];
rz(1.4615116419614171) q[10];
ry(1.5789136729870943) q[11];
rz(-0.18484967176593384) q[11];
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
ry(2.9633581488554257) q[0];
rz(-2.814847339095467) q[0];
ry(-1.5945011261059854) q[1];
rz(2.3354476334620964) q[1];
ry(0.0008542062448931867) q[2];
rz(-2.144207289835342) q[2];
ry(-3.0891035472707817) q[3];
rz(0.6748528942919368) q[3];
ry(-1.5720967881592909) q[4];
rz(-1.7174077239964993) q[4];
ry(-1.2286322851658893) q[5];
rz(-3.0464793773652423) q[5];
ry(0.7804548779773101) q[6];
rz(2.7434816226066765) q[6];
ry(-2.5900108691579886) q[7];
rz(2.182308271187072) q[7];
ry(-1.7433171760564479) q[8];
rz(-2.3219924342147875) q[8];
ry(2.77430812604838) q[9];
rz(0.47291848973470346) q[9];
ry(-1.568481541556154) q[10];
rz(-1.494594302452441) q[10];
ry(-1.8378540492133402) q[11];
rz(-2.613056655964619) q[11];