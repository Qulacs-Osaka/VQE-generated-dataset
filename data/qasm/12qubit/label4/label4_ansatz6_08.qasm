OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.3932010099043469) q[0];
ry(1.420717567284516) q[1];
cx q[0],q[1];
ry(0.10255807763953051) q[0];
ry(3.060490509899352) q[1];
cx q[0],q[1];
ry(2.5478890967900694) q[1];
ry(0.07687892498434756) q[2];
cx q[1],q[2];
ry(1.2933433432813353) q[1];
ry(2.809013103588838) q[2];
cx q[1],q[2];
ry(2.807042201959877) q[2];
ry(-0.9243600874221086) q[3];
cx q[2],q[3];
ry(-2.8042279290431575) q[2];
ry(-2.610437848288181) q[3];
cx q[2],q[3];
ry(-1.3623422517091122) q[3];
ry(-1.2779479506127176) q[4];
cx q[3],q[4];
ry(0.44284045344161793) q[3];
ry(1.0006308474695966) q[4];
cx q[3],q[4];
ry(-0.19356636967825294) q[4];
ry(-1.5372709623251897) q[5];
cx q[4],q[5];
ry(-0.8769432703947735) q[4];
ry(-3.113411484935918) q[5];
cx q[4],q[5];
ry(-2.3634862363003384) q[5];
ry(0.6398792958581865) q[6];
cx q[5],q[6];
ry(-3.0101575578104813) q[5];
ry(-0.6759921308779075) q[6];
cx q[5],q[6];
ry(-0.1519773792837908) q[6];
ry(-2.122635372150654) q[7];
cx q[6],q[7];
ry(2.240264979532158) q[6];
ry(1.5421641666269386) q[7];
cx q[6],q[7];
ry(0.24891352502101535) q[7];
ry(1.4442065184904198) q[8];
cx q[7],q[8];
ry(-1.3643050526257765) q[7];
ry(-2.987958465922749) q[8];
cx q[7],q[8];
ry(1.9288732294468958) q[8];
ry(-0.5479543944024607) q[9];
cx q[8],q[9];
ry(-1.9270391768850503) q[8];
ry(-1.1010238091935773) q[9];
cx q[8],q[9];
ry(2.369742006120558) q[9];
ry(0.386347892371988) q[10];
cx q[9],q[10];
ry(2.0714894172254885) q[9];
ry(2.1671028335110964) q[10];
cx q[9],q[10];
ry(-2.4604030312943537) q[10];
ry(0.5985181818481847) q[11];
cx q[10],q[11];
ry(-2.482679177363079) q[10];
ry(-0.42667659375186123) q[11];
cx q[10],q[11];
ry(1.2641107231274331) q[0];
ry(1.8964631913265115) q[1];
cx q[0],q[1];
ry(-2.9956456176262316) q[0];
ry(0.2938546072061792) q[1];
cx q[0],q[1];
ry(-0.7677126626947809) q[1];
ry(1.2256125172443328) q[2];
cx q[1],q[2];
ry(2.0976251395096877) q[1];
ry(1.108719927566498) q[2];
cx q[1],q[2];
ry(1.987554600433855) q[2];
ry(-2.330829224543646) q[3];
cx q[2],q[3];
ry(0.8932282953193185) q[2];
ry(-0.06571751486389724) q[3];
cx q[2],q[3];
ry(-2.8785387139769796) q[3];
ry(0.6372393487171876) q[4];
cx q[3],q[4];
ry(0.7555275996992927) q[3];
ry(3.0669760808348516) q[4];
cx q[3],q[4];
ry(-1.800758009236893) q[4];
ry(2.706504783534689) q[5];
cx q[4],q[5];
ry(0.12683631323055966) q[4];
ry(0.31829935486736244) q[5];
cx q[4],q[5];
ry(-0.3441240747307661) q[5];
ry(0.44215127594020665) q[6];
cx q[5],q[6];
ry(-2.7178992837031544) q[5];
ry(1.8302792768792118) q[6];
cx q[5],q[6];
ry(-3.1260754614869035) q[6];
ry(-1.412234066486406) q[7];
cx q[6],q[7];
ry(1.9328001772128918) q[6];
ry(-2.7881506280736463) q[7];
cx q[6],q[7];
ry(-1.8565723207866318) q[7];
ry(1.501192148181414) q[8];
cx q[7],q[8];
ry(-1.2342824781678843) q[7];
ry(1.1469371628297853) q[8];
cx q[7],q[8];
ry(2.9490907449978123) q[8];
ry(-2.147703351243072) q[9];
cx q[8],q[9];
ry(3.0533118871168536) q[8];
ry(3.0550231744560055) q[9];
cx q[8],q[9];
ry(-0.615335137327689) q[9];
ry(2.860914859451813) q[10];
cx q[9],q[10];
ry(1.0300976915355413) q[9];
ry(0.7646226272678647) q[10];
cx q[9],q[10];
ry(-2.066514737026771) q[10];
ry(0.1948940132545065) q[11];
cx q[10],q[11];
ry(-0.058264013179510764) q[10];
ry(1.7012372593539113) q[11];
cx q[10],q[11];
ry(2.744468274449905) q[0];
ry(1.5960936210603593) q[1];
cx q[0],q[1];
ry(1.1704379240977996) q[0];
ry(0.4784940007459105) q[1];
cx q[0],q[1];
ry(-1.4297722601116307) q[1];
ry(-1.8766716342056464) q[2];
cx q[1],q[2];
ry(0.00784092112117504) q[1];
ry(-1.976353491105338) q[2];
cx q[1],q[2];
ry(-1.4648060545071644) q[2];
ry(-2.6197350674559634) q[3];
cx q[2],q[3];
ry(-1.1632507186000307) q[2];
ry(2.841198894937863) q[3];
cx q[2],q[3];
ry(1.8689842493994135) q[3];
ry(2.8696030729627373) q[4];
cx q[3],q[4];
ry(0.07157063901537775) q[3];
ry(2.5907052562581145) q[4];
cx q[3],q[4];
ry(1.174123175758315) q[4];
ry(1.60686756976605) q[5];
cx q[4],q[5];
ry(-0.2392615573207717) q[4];
ry(-3.089881991309167) q[5];
cx q[4],q[5];
ry(-0.9667479990595346) q[5];
ry(-1.7427480090120169) q[6];
cx q[5],q[6];
ry(-1.761037782961353) q[5];
ry(2.678788440312953) q[6];
cx q[5],q[6];
ry(0.004573510798822771) q[6];
ry(2.9964354187791282) q[7];
cx q[6],q[7];
ry(1.645191619313529) q[6];
ry(-0.008996837869352379) q[7];
cx q[6],q[7];
ry(-2.4507086502498034) q[7];
ry(-1.6509527173932002) q[8];
cx q[7],q[8];
ry(-0.09172611837471983) q[7];
ry(-0.16202958902496434) q[8];
cx q[7],q[8];
ry(-3.0860034581816485) q[8];
ry(-3.067401465616589) q[9];
cx q[8],q[9];
ry(-2.5835427682325207) q[8];
ry(2.916626422042589) q[9];
cx q[8],q[9];
ry(1.6007074321528976) q[9];
ry(0.32684884341860077) q[10];
cx q[9],q[10];
ry(-3.106588810278402) q[9];
ry(2.883899263935274) q[10];
cx q[9],q[10];
ry(2.1691312778242713) q[10];
ry(0.8008884783813383) q[11];
cx q[10],q[11];
ry(1.501260759142851) q[10];
ry(0.10234075638131943) q[11];
cx q[10],q[11];
ry(2.5896397659302672) q[0];
ry(1.7069657867165826) q[1];
cx q[0],q[1];
ry(-1.0322427504070157) q[0];
ry(-1.7436888465651732) q[1];
cx q[0],q[1];
ry(-0.5580506796353415) q[1];
ry(-3.0822776270466203) q[2];
cx q[1],q[2];
ry(0.7591344420511864) q[1];
ry(-0.35419956792978713) q[2];
cx q[1],q[2];
ry(-2.527858943184243) q[2];
ry(0.404019096736296) q[3];
cx q[2],q[3];
ry(-1.2565630661194875) q[2];
ry(1.5524845204514293) q[3];
cx q[2],q[3];
ry(-0.7326171677900994) q[3];
ry(0.5426259408280675) q[4];
cx q[3],q[4];
ry(0.42996582598395056) q[3];
ry(-2.377695822571251) q[4];
cx q[3],q[4];
ry(-1.7907051317749811) q[4];
ry(-2.7630915787442074) q[5];
cx q[4],q[5];
ry(0.08649900364448973) q[4];
ry(-1.6626446223832838) q[5];
cx q[4],q[5];
ry(-2.8048005963775737) q[5];
ry(1.0755227677386658) q[6];
cx q[5],q[6];
ry(-3.057111761182127) q[5];
ry(2.0098996001297493) q[6];
cx q[5],q[6];
ry(-0.7782205775295576) q[6];
ry(-0.5694587971815436) q[7];
cx q[6],q[7];
ry(-1.457251214167585) q[6];
ry(-1.3812347895809296) q[7];
cx q[6],q[7];
ry(2.769659153672039) q[7];
ry(-1.1660229091659442) q[8];
cx q[7],q[8];
ry(2.591376367131589) q[7];
ry(-2.997253054349535) q[8];
cx q[7],q[8];
ry(-1.4459622681500903) q[8];
ry(-1.1750699071884867) q[9];
cx q[8],q[9];
ry(2.7883455181867745) q[8];
ry(-2.0667292160024195) q[9];
cx q[8],q[9];
ry(-0.6966697441235465) q[9];
ry(2.891812892112017) q[10];
cx q[9],q[10];
ry(-1.8107082751900505) q[9];
ry(1.6495738134217568) q[10];
cx q[9],q[10];
ry(-2.7105268748780413) q[10];
ry(-1.9838095346655864) q[11];
cx q[10],q[11];
ry(0.043064475581390066) q[10];
ry(2.9258024325541716) q[11];
cx q[10],q[11];
ry(1.4833816911347664) q[0];
ry(1.6913077910099625) q[1];
cx q[0],q[1];
ry(-1.5023625282183664) q[0];
ry(0.24758637128929983) q[1];
cx q[0],q[1];
ry(1.2615828954693655) q[1];
ry(2.9115929890922603) q[2];
cx q[1],q[2];
ry(1.491687372643197) q[1];
ry(-1.5381950887586608) q[2];
cx q[1],q[2];
ry(0.35217213142608017) q[2];
ry(0.38682429326108597) q[3];
cx q[2],q[3];
ry(0.06987549619034895) q[2];
ry(-2.906595411679205) q[3];
cx q[2],q[3];
ry(2.3089045020900896) q[3];
ry(-1.8583369477581588) q[4];
cx q[3],q[4];
ry(-0.06430662931915501) q[3];
ry(0.03190531634372373) q[4];
cx q[3],q[4];
ry(-0.010025772784685751) q[4];
ry(-1.6578873256052762) q[5];
cx q[4],q[5];
ry(-1.8586221680071144) q[4];
ry(3.0354597213272507) q[5];
cx q[4],q[5];
ry(-1.7091282377869161) q[5];
ry(2.658384804164635) q[6];
cx q[5],q[6];
ry(0.1364136562008701) q[5];
ry(-1.7905171743257908) q[6];
cx q[5],q[6];
ry(-2.5804222253369784) q[6];
ry(1.103080851308367) q[7];
cx q[6],q[7];
ry(-1.6793568878951217) q[6];
ry(2.7654466388338546) q[7];
cx q[6],q[7];
ry(-1.5927046763400017) q[7];
ry(-1.8452930797620857) q[8];
cx q[7],q[8];
ry(0.03857364846353039) q[7];
ry(2.595523081883646) q[8];
cx q[7],q[8];
ry(1.8520038069336282) q[8];
ry(1.6045493421611918) q[9];
cx q[8],q[9];
ry(0.6614769463675536) q[8];
ry(1.594471968582248) q[9];
cx q[8],q[9];
ry(-0.9017487637397757) q[9];
ry(-0.008623140942639829) q[10];
cx q[9],q[10];
ry(1.224000375101712) q[9];
ry(2.914143821286811) q[10];
cx q[9],q[10];
ry(2.9585598632435515) q[10];
ry(-1.9069309598847715) q[11];
cx q[10],q[11];
ry(1.149900199349887) q[10];
ry(-2.7827137921041785) q[11];
cx q[10],q[11];
ry(2.8532678539886214) q[0];
ry(0.5423749150118704) q[1];
cx q[0],q[1];
ry(0.48597074258026574) q[0];
ry(-2.7159528404968065) q[1];
cx q[0],q[1];
ry(0.12348022662449777) q[1];
ry(-0.6415363393408511) q[2];
cx q[1],q[2];
ry(-0.00785955814367778) q[1];
ry(-0.48329229424962966) q[2];
cx q[1],q[2];
ry(2.203141592099077) q[2];
ry(0.4401360817006223) q[3];
cx q[2],q[3];
ry(0.030423934328946647) q[2];
ry(-0.6630504536085494) q[3];
cx q[2],q[3];
ry(2.507685437133474) q[3];
ry(0.859561842356191) q[4];
cx q[3],q[4];
ry(-3.069083261229079) q[3];
ry(-0.09166309502558659) q[4];
cx q[3],q[4];
ry(0.7878332474728773) q[4];
ry(2.4583798621356943) q[5];
cx q[4],q[5];
ry(-0.1840715379207989) q[4];
ry(-0.02388151917991532) q[5];
cx q[4],q[5];
ry(-3.129758140012554) q[5];
ry(1.643363183141016) q[6];
cx q[5],q[6];
ry(0.3930516487242892) q[5];
ry(-0.8072519312821184) q[6];
cx q[5],q[6];
ry(0.7579268821774887) q[6];
ry(1.567388647193292) q[7];
cx q[6],q[7];
ry(1.5835904250946728) q[6];
ry(0.00487023503261297) q[7];
cx q[6],q[7];
ry(0.6879203992866806) q[7];
ry(-0.9910210405214626) q[8];
cx q[7],q[8];
ry(1.4464323181012855) q[7];
ry(-0.0139812875147598) q[8];
cx q[7],q[8];
ry(-0.7511302655730381) q[8];
ry(-1.945584279261947) q[9];
cx q[8],q[9];
ry(-0.03772625949722994) q[8];
ry(0.0018291108088735902) q[9];
cx q[8],q[9];
ry(-0.5370125667079644) q[9];
ry(-0.5192692185648895) q[10];
cx q[9],q[10];
ry(0.21712774384188038) q[9];
ry(1.9442100501677038) q[10];
cx q[9],q[10];
ry(-1.2528908948376074) q[10];
ry(0.9788140597042324) q[11];
cx q[10],q[11];
ry(1.6798145590434213) q[10];
ry(-1.6170195658168507) q[11];
cx q[10],q[11];
ry(-0.8403371337280429) q[0];
ry(1.7672826522090297) q[1];
cx q[0],q[1];
ry(0.5511173461119624) q[0];
ry(-0.2812492890234761) q[1];
cx q[0],q[1];
ry(0.4635772014889912) q[1];
ry(-0.2574892116540132) q[2];
cx q[1],q[2];
ry(-1.1876741056004598) q[1];
ry(-2.1029682718503873) q[2];
cx q[1],q[2];
ry(-2.220801432972819) q[2];
ry(0.3976319175722377) q[3];
cx q[2],q[3];
ry(0.025888839995568347) q[2];
ry(3.0363125028052242) q[3];
cx q[2],q[3];
ry(3.068504367979797) q[3];
ry(-2.8099914705404165) q[4];
cx q[3],q[4];
ry(2.6638094437695523) q[3];
ry(-1.4623020767853943) q[4];
cx q[3],q[4];
ry(-2.9893596072233537) q[4];
ry(0.05057166665673396) q[5];
cx q[4],q[5];
ry(3.1041600250149886) q[4];
ry(0.2545940189413072) q[5];
cx q[4],q[5];
ry(1.1720351519682586) q[5];
ry(-1.806656100724874) q[6];
cx q[5],q[6];
ry(-3.0581558039167596) q[5];
ry(-0.3233781504035572) q[6];
cx q[5],q[6];
ry(0.6005899088150405) q[6];
ry(-0.9994476308191284) q[7];
cx q[6],q[7];
ry(3.126711871346887) q[6];
ry(2.9376079358563527) q[7];
cx q[6],q[7];
ry(1.2371578989524734) q[7];
ry(-1.8367006806737136) q[8];
cx q[7],q[8];
ry(-2.5996430303503906) q[7];
ry(-1.6386256145643023) q[8];
cx q[7],q[8];
ry(0.6671288667555704) q[8];
ry(-1.1220805058560739) q[9];
cx q[8],q[9];
ry(0.021132542907101204) q[8];
ry(3.091149500396995) q[9];
cx q[8],q[9];
ry(2.241916699330367) q[9];
ry(2.0781977315321036) q[10];
cx q[9],q[10];
ry(1.164930583848536) q[9];
ry(1.2642277152023205) q[10];
cx q[9],q[10];
ry(2.906052423125647) q[10];
ry(2.864172845628076) q[11];
cx q[10],q[11];
ry(0.22249291218409917) q[10];
ry(-2.8244229075514884) q[11];
cx q[10],q[11];
ry(-1.8486919828709543) q[0];
ry(-2.945819613403999) q[1];
cx q[0],q[1];
ry(-0.11680614237509701) q[0];
ry(2.97449290238375) q[1];
cx q[0],q[1];
ry(-1.0639842657231064) q[1];
ry(-2.098001530721074) q[2];
cx q[1],q[2];
ry(-2.890533880600228) q[1];
ry(-2.4853465493183977) q[2];
cx q[1],q[2];
ry(-1.482264947625789) q[2];
ry(1.5588239836414308) q[3];
cx q[2],q[3];
ry(1.692152159247664) q[2];
ry(0.017627080118444383) q[3];
cx q[2],q[3];
ry(-0.18811927552648644) q[3];
ry(0.6034384183200572) q[4];
cx q[3],q[4];
ry(0.00048327807939126295) q[3];
ry(-0.003970897195054945) q[4];
cx q[3],q[4];
ry(-2.193640849626154) q[4];
ry(3.0287087982567535) q[5];
cx q[4],q[5];
ry(-1.6779767274516635) q[4];
ry(1.6900656605615827) q[5];
cx q[4],q[5];
ry(1.5163426408498815) q[5];
ry(-1.4477590366620685) q[6];
cx q[5],q[6];
ry(1.2071899685237086) q[5];
ry(1.5317147933294188) q[6];
cx q[5],q[6];
ry(-1.6116052271626393) q[6];
ry(2.847016795206489) q[7];
cx q[6],q[7];
ry(3.109585596494452) q[6];
ry(-0.16421585300589966) q[7];
cx q[6],q[7];
ry(-2.3299130751183603) q[7];
ry(-3.0755598748150486) q[8];
cx q[7],q[8];
ry(2.3599023119045808) q[7];
ry(1.5431987505750577) q[8];
cx q[7],q[8];
ry(2.818465095451409) q[8];
ry(-2.635798822677732) q[9];
cx q[8],q[9];
ry(-3.11253461253092) q[8];
ry(3.116057112539665) q[9];
cx q[8],q[9];
ry(1.7992418029224069) q[9];
ry(-2.855449231881465) q[10];
cx q[9],q[10];
ry(0.7993222510536173) q[9];
ry(1.1132049272115125) q[10];
cx q[9],q[10];
ry(-1.8639043465920735) q[10];
ry(-0.6290956391829581) q[11];
cx q[10],q[11];
ry(-1.3224694954289298) q[10];
ry(-1.7752106116720257) q[11];
cx q[10],q[11];
ry(-2.788397774234235) q[0];
ry(-1.8807974039653355) q[1];
cx q[0],q[1];
ry(-3.100150987361011) q[0];
ry(-0.5759828646408129) q[1];
cx q[0],q[1];
ry(-2.1238388236740997) q[1];
ry(-1.3994156413287318) q[2];
cx q[1],q[2];
ry(1.6179678850127004) q[1];
ry(0.6108647148393288) q[2];
cx q[1],q[2];
ry(0.15667734838693503) q[2];
ry(-2.9494752183138786) q[3];
cx q[2],q[3];
ry(1.4712136389973731) q[2];
ry(-3.139487401237458) q[3];
cx q[2],q[3];
ry(-1.57120173820368) q[3];
ry(-1.3208029193838857) q[4];
cx q[3],q[4];
ry(-1.5707751159884202) q[3];
ry(-1.8291485431131864) q[4];
cx q[3],q[4];
ry(-2.076015942344535) q[4];
ry(1.5744192884803037) q[5];
cx q[4],q[5];
ry(1.435249907748644) q[4];
ry(-1.3217458195585723) q[5];
cx q[4],q[5];
ry(0.3533612295041193) q[5];
ry(-0.9307167622717294) q[6];
cx q[5],q[6];
ry(-3.0389226320563663) q[5];
ry(0.018681565061090675) q[6];
cx q[5],q[6];
ry(1.5478523913414834) q[6];
ry(0.9039551943481383) q[7];
cx q[6],q[7];
ry(0.004854319173582996) q[6];
ry(-0.0021190790152945615) q[7];
cx q[6],q[7];
ry(1.1567993630474886) q[7];
ry(-2.499518238115877) q[8];
cx q[7],q[8];
ry(-2.021974555771568) q[7];
ry(1.1567658094198074) q[8];
cx q[7],q[8];
ry(-0.8784812302447379) q[8];
ry(-2.697016992239477) q[9];
cx q[8],q[9];
ry(-0.7062690122561983) q[8];
ry(-1.602161886272163) q[9];
cx q[8],q[9];
ry(-0.24406921376827903) q[9];
ry(1.802032316704801) q[10];
cx q[9],q[10];
ry(0.5464204915859936) q[9];
ry(-3.140508190675766) q[10];
cx q[9],q[10];
ry(1.3437388626357434) q[10];
ry(0.39459307323589243) q[11];
cx q[10],q[11];
ry(0.4640810973289139) q[10];
ry(-1.010703951676435) q[11];
cx q[10],q[11];
ry(2.6824867707234747) q[0];
ry(-2.158678908810702) q[1];
cx q[0],q[1];
ry(0.23765524961056972) q[0];
ry(-0.857344296188562) q[1];
cx q[0],q[1];
ry(2.4169197284881765) q[1];
ry(1.5879173576662131) q[2];
cx q[1],q[2];
ry(2.2396366321886605) q[1];
ry(-0.144032338114298) q[2];
cx q[1],q[2];
ry(2.8700716320672184) q[2];
ry(1.3254597529221308) q[3];
cx q[2],q[3];
ry(0.020428985151932366) q[2];
ry(0.07637517149642203) q[3];
cx q[2],q[3];
ry(-2.817618295693364) q[3];
ry(2.059275279517151) q[4];
cx q[3],q[4];
ry(0.0001858130990401297) q[3];
ry(0.017673790401101726) q[4];
cx q[3],q[4];
ry(-0.4511724319966959) q[4];
ry(2.7742924445759156) q[5];
cx q[4],q[5];
ry(0.12841356263575107) q[4];
ry(2.6509565825943624) q[5];
cx q[4],q[5];
ry(1.5853498958688) q[5];
ry(2.221567362724425) q[6];
cx q[5],q[6];
ry(-1.4581193113017443) q[5];
ry(1.51215145540238) q[6];
cx q[5],q[6];
ry(-0.454256807950566) q[6];
ry(2.8839542884350133) q[7];
cx q[6],q[7];
ry(-1.5947596823833121) q[6];
ry(-3.0956450849699286) q[7];
cx q[6],q[7];
ry(0.6500615526116674) q[7];
ry(1.5361457432617012) q[8];
cx q[7],q[8];
ry(2.230989361418394) q[7];
ry(3.1412632589023732) q[8];
cx q[7],q[8];
ry(-1.570462783101867) q[8];
ry(2.791452460010422) q[9];
cx q[8],q[9];
ry(0.0007848997377690736) q[8];
ry(-1.4943076747487822) q[9];
cx q[8],q[9];
ry(2.468368026027784) q[9];
ry(-1.411227368978829) q[10];
cx q[9],q[10];
ry(0.6962965032411566) q[9];
ry(-0.023607751764853004) q[10];
cx q[9],q[10];
ry(0.7023954681708717) q[10];
ry(1.0886314156151418) q[11];
cx q[10],q[11];
ry(1.4543682750423044) q[10];
ry(-2.8210655537051346) q[11];
cx q[10],q[11];
ry(2.3094394694165343) q[0];
ry(2.228588862840856) q[1];
cx q[0],q[1];
ry(1.4470933622972815) q[0];
ry(1.6741957250617183) q[1];
cx q[0],q[1];
ry(-1.7210587141163969) q[1];
ry(-1.5748898008638206) q[2];
cx q[1],q[2];
ry(-2.8888808688375125) q[1];
ry(1.584952126129073) q[2];
cx q[1],q[2];
ry(-0.09186706610422668) q[2];
ry(-0.03681229504971028) q[3];
cx q[2],q[3];
ry(-0.05549482570377196) q[2];
ry(-0.0012591235660153188) q[3];
cx q[2],q[3];
ry(-1.6083620479881757) q[3];
ry(3.051349704455762) q[4];
cx q[3],q[4];
ry(-0.09261131587969675) q[3];
ry(-1.8272039832839095) q[4];
cx q[3],q[4];
ry(1.5736682854137796) q[4];
ry(-1.5864630057952809) q[5];
cx q[4],q[5];
ry(-1.3042577373586637) q[4];
ry(-1.439690521495784) q[5];
cx q[4],q[5];
ry(2.1095858876533002) q[5];
ry(-0.20256763214954834) q[6];
cx q[5],q[6];
ry(-3.13070976002374) q[5];
ry(3.1402375036366856) q[6];
cx q[5],q[6];
ry(-2.327416994045255) q[6];
ry(0.6700331974263358) q[7];
cx q[6],q[7];
ry(3.098714105030722) q[6];
ry(3.088615553296792) q[7];
cx q[6],q[7];
ry(-1.5514130419195054) q[7];
ry(-2.0069640359034473) q[8];
cx q[7],q[8];
ry(2.431948831008598) q[7];
ry(2.7670862126776634) q[8];
cx q[7],q[8];
ry(3.0728646622567792) q[8];
ry(0.8400340903421099) q[9];
cx q[8],q[9];
ry(-1.5781666766343818) q[8];
ry(-3.1411667799743452) q[9];
cx q[8],q[9];
ry(2.506312500159455) q[9];
ry(-2.5417335164168415) q[10];
cx q[9],q[10];
ry(1.5708652989861214) q[9];
ry(3.1414747756889416) q[10];
cx q[9],q[10];
ry(-1.5710018864512392) q[10];
ry(1.1860373307700742) q[11];
cx q[10],q[11];
ry(1.5696146690248751) q[10];
ry(2.3770336235536385) q[11];
cx q[10],q[11];
ry(-1.2559395950458563) q[0];
ry(-1.567988568567137) q[1];
ry(3.1336562779677326) q[2];
ry(1.583091946671714) q[3];
ry(-1.5722159734476877) q[4];
ry(-1.0321806411665668) q[5];
ry(0.6225352663347125) q[6];
ry(-1.5705568004841561) q[7];
ry(-2.8700505782411003) q[8];
ry(-0.6353716499777028) q[9];
ry(1.5703295696046036) q[10];
ry(-1.571192295122071) q[11];