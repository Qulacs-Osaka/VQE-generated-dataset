OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.8298083407510219) q[0];
ry(-1.228377316927417) q[1];
cx q[0],q[1];
ry(-0.25370347218249634) q[0];
ry(-3.1126996415820694) q[1];
cx q[0],q[1];
ry(1.1557405655678155) q[2];
ry(-2.471838790356123) q[3];
cx q[2],q[3];
ry(-0.9936025900455463) q[2];
ry(-2.94154334335621) q[3];
cx q[2],q[3];
ry(2.9275294043607256) q[0];
ry(-0.5626186139412509) q[2];
cx q[0],q[2];
ry(2.3328326034524) q[0];
ry(-0.03514560761724879) q[2];
cx q[0],q[2];
ry(-3.045471982791483) q[1];
ry(0.7462120654219486) q[3];
cx q[1],q[3];
ry(2.0685251890741183) q[1];
ry(-1.6578632864352854) q[3];
cx q[1],q[3];
ry(2.022605787019743) q[0];
ry(0.3316833114958673) q[1];
cx q[0],q[1];
ry(-1.0555544698045336) q[0];
ry(-0.605556170269928) q[1];
cx q[0],q[1];
ry(-0.30164177033350986) q[2];
ry(-2.8960850243491536) q[3];
cx q[2],q[3];
ry(-1.3036658803766714) q[2];
ry(2.2799948657826454) q[3];
cx q[2],q[3];
ry(2.418581758358913) q[0];
ry(-1.1286389318431576) q[2];
cx q[0],q[2];
ry(-0.22091526272964082) q[0];
ry(-0.0905465134897072) q[2];
cx q[0],q[2];
ry(0.6728424149897343) q[1];
ry(-2.6470078852577985) q[3];
cx q[1],q[3];
ry(0.27991187213585533) q[1];
ry(-2.5199194460871372) q[3];
cx q[1],q[3];
ry(0.8403270527682163) q[0];
ry(2.75406034233313) q[1];
cx q[0],q[1];
ry(-1.6621600948274464) q[0];
ry(2.66476104631636) q[1];
cx q[0],q[1];
ry(2.025563869434488) q[2];
ry(-2.028095730429472) q[3];
cx q[2],q[3];
ry(-0.035329875157781056) q[2];
ry(-2.912627173748078) q[3];
cx q[2],q[3];
ry(0.21033791223512052) q[0];
ry(-3.032921824906549) q[2];
cx q[0],q[2];
ry(-1.1336514288970436) q[0];
ry(-2.2194763044339725) q[2];
cx q[0],q[2];
ry(-1.3885865014569474) q[1];
ry(1.8469479157984168) q[3];
cx q[1],q[3];
ry(3.0514834040176377) q[1];
ry(-0.6241240421146796) q[3];
cx q[1],q[3];
ry(1.2138779343368116) q[0];
ry(-1.4622191914680505) q[1];
cx q[0],q[1];
ry(2.620381219687653) q[0];
ry(1.748951184632415) q[1];
cx q[0],q[1];
ry(-1.7288047986376522) q[2];
ry(0.1282960839113681) q[3];
cx q[2],q[3];
ry(-0.9933719382096466) q[2];
ry(3.007032580854652) q[3];
cx q[2],q[3];
ry(0.6385035130328278) q[0];
ry(-1.4230724307700653) q[2];
cx q[0],q[2];
ry(-2.5974715292757944) q[0];
ry(1.7974865738363752) q[2];
cx q[0],q[2];
ry(-2.000668991736011) q[1];
ry(-1.8308313255582649) q[3];
cx q[1],q[3];
ry(1.1706615045011755) q[1];
ry(-2.2125071277330033) q[3];
cx q[1],q[3];
ry(-0.5234798280111292) q[0];
ry(-0.06588070527964263) q[1];
cx q[0],q[1];
ry(0.8197470740421221) q[0];
ry(1.268978490254231) q[1];
cx q[0],q[1];
ry(1.352048026079106) q[2];
ry(-0.49069781180699923) q[3];
cx q[2],q[3];
ry(2.788646365465196) q[2];
ry(2.127732928461077) q[3];
cx q[2],q[3];
ry(0.5116536164635441) q[0];
ry(-1.5030692806209451) q[2];
cx q[0],q[2];
ry(-0.8309054180094799) q[0];
ry(-0.4476903796031966) q[2];
cx q[0],q[2];
ry(-2.0504608646339753) q[1];
ry(-2.98181643008768) q[3];
cx q[1],q[3];
ry(-2.337569567555877) q[1];
ry(3.081258396759281) q[3];
cx q[1],q[3];
ry(1.860263914962664) q[0];
ry(-2.982988153381936) q[1];
cx q[0],q[1];
ry(2.9335486882727912) q[0];
ry(-2.2132633184928023) q[1];
cx q[0],q[1];
ry(-1.2840871800951161) q[2];
ry(-2.896952566067822) q[3];
cx q[2],q[3];
ry(-0.8819433125984855) q[2];
ry(-0.09343110258731536) q[3];
cx q[2],q[3];
ry(2.0220684184873603) q[0];
ry(-0.23169058626618536) q[2];
cx q[0],q[2];
ry(0.09794592687225467) q[0];
ry(-2.263828517387546) q[2];
cx q[0],q[2];
ry(2.29030009921846) q[1];
ry(2.908770550839644) q[3];
cx q[1],q[3];
ry(0.8090126666598181) q[1];
ry(-0.5093468133675715) q[3];
cx q[1],q[3];
ry(0.15428070429763577) q[0];
ry(1.3956466482516472) q[1];
cx q[0],q[1];
ry(-0.5294316731474183) q[0];
ry(0.5845957156920214) q[1];
cx q[0],q[1];
ry(-1.0960570030374381) q[2];
ry(-1.087531371938617) q[3];
cx q[2],q[3];
ry(-0.9516477707031067) q[2];
ry(-1.024341868441006) q[3];
cx q[2],q[3];
ry(-0.8830971239061364) q[0];
ry(-2.0826369475522117) q[2];
cx q[0],q[2];
ry(-0.0486845048049634) q[0];
ry(2.174902551693072) q[2];
cx q[0],q[2];
ry(0.15093352146331768) q[1];
ry(1.820112780863183) q[3];
cx q[1],q[3];
ry(-1.1480799550945608) q[1];
ry(-2.257027212276041) q[3];
cx q[1],q[3];
ry(-2.2655327596161214) q[0];
ry(2.7745758293635845) q[1];
cx q[0],q[1];
ry(2.6750122903290907) q[0];
ry(0.030712717692514397) q[1];
cx q[0],q[1];
ry(-1.9396701208332914) q[2];
ry(-2.442097340409368) q[3];
cx q[2],q[3];
ry(0.45969918180677727) q[2];
ry(-0.5053217816877105) q[3];
cx q[2],q[3];
ry(-2.0769146134436998) q[0];
ry(-2.683829559804215) q[2];
cx q[0],q[2];
ry(-3.030072768152616) q[0];
ry(2.3046792064638724) q[2];
cx q[0],q[2];
ry(-1.0193134504848695) q[1];
ry(2.327759223224691) q[3];
cx q[1],q[3];
ry(0.2812281016683601) q[1];
ry(-2.9962126913644287) q[3];
cx q[1],q[3];
ry(0.39809685522486915) q[0];
ry(2.3112595141406596) q[1];
cx q[0],q[1];
ry(-0.2968316981561179) q[0];
ry(3.0898234017607007) q[1];
cx q[0],q[1];
ry(-0.0925219822832966) q[2];
ry(0.17478395699812554) q[3];
cx q[2],q[3];
ry(1.049264956595633) q[2];
ry(1.9087714894425174) q[3];
cx q[2],q[3];
ry(2.166538722666468) q[0];
ry(0.43020584835802644) q[2];
cx q[0],q[2];
ry(2.6668078792060155) q[0];
ry(-2.8235988350383474) q[2];
cx q[0],q[2];
ry(2.2161467745605106) q[1];
ry(-1.7127268146591785) q[3];
cx q[1],q[3];
ry(0.4499467598971636) q[1];
ry(-2.7299417952840472) q[3];
cx q[1],q[3];
ry(-1.1048524657164698) q[0];
ry(-0.34169698997447895) q[1];
cx q[0],q[1];
ry(-2.6709088165502117) q[0];
ry(-2.302782488606357) q[1];
cx q[0],q[1];
ry(-1.4445029643340597) q[2];
ry(-2.83318733763452) q[3];
cx q[2],q[3];
ry(1.7420405024208243) q[2];
ry(0.6983518189781474) q[3];
cx q[2],q[3];
ry(-1.97705854688751) q[0];
ry(-0.2164857307388739) q[2];
cx q[0],q[2];
ry(0.9156743027349655) q[0];
ry(1.1315385636156539) q[2];
cx q[0],q[2];
ry(-0.13213641000309817) q[1];
ry(-2.0791079077078454) q[3];
cx q[1],q[3];
ry(-1.984867595869928) q[1];
ry(-1.4973874980056845) q[3];
cx q[1],q[3];
ry(-1.1122486538838745) q[0];
ry(0.5633222357726378) q[1];
cx q[0],q[1];
ry(-1.1910593270485037) q[0];
ry(-2.3075084352123727) q[1];
cx q[0],q[1];
ry(-0.5981942609917855) q[2];
ry(-1.4109601204283244) q[3];
cx q[2],q[3];
ry(2.601930288282243) q[2];
ry(-2.1192548369181416) q[3];
cx q[2],q[3];
ry(2.713396857228692) q[0];
ry(2.1553483308934087) q[2];
cx q[0],q[2];
ry(-0.38891580817709315) q[0];
ry(0.47840985689821985) q[2];
cx q[0],q[2];
ry(-0.5745874574498098) q[1];
ry(-2.4949515251692076) q[3];
cx q[1],q[3];
ry(2.4023215743820248) q[1];
ry(-3.0635260974907212) q[3];
cx q[1],q[3];
ry(1.294744640733033) q[0];
ry(-1.2439499838892152) q[1];
cx q[0],q[1];
ry(-0.3284195042626258) q[0];
ry(-0.9173120280885285) q[1];
cx q[0],q[1];
ry(1.60639923177404) q[2];
ry(0.5259777189440564) q[3];
cx q[2],q[3];
ry(1.675265861491481) q[2];
ry(0.9097030972472713) q[3];
cx q[2],q[3];
ry(-1.8409405624777033) q[0];
ry(-0.9357442926797912) q[2];
cx q[0],q[2];
ry(-0.13602643841217785) q[0];
ry(3.050268214866415) q[2];
cx q[0],q[2];
ry(-2.2307618166352583) q[1];
ry(-0.9671172880065706) q[3];
cx q[1],q[3];
ry(-0.013343171431394296) q[1];
ry(-0.9307506587623297) q[3];
cx q[1],q[3];
ry(1.0249869629343382) q[0];
ry(-2.85918537513157) q[1];
cx q[0],q[1];
ry(-0.5722399063397315) q[0];
ry(-1.7530515569837213) q[1];
cx q[0],q[1];
ry(0.5125660821112697) q[2];
ry(-1.9802860413904384) q[3];
cx q[2],q[3];
ry(2.65964596452736) q[2];
ry(2.791429927749971) q[3];
cx q[2],q[3];
ry(1.6428519940134574) q[0];
ry(0.04507482394432305) q[2];
cx q[0],q[2];
ry(-1.9427085022458928) q[0];
ry(-2.022398309815183) q[2];
cx q[0],q[2];
ry(0.046184886112465406) q[1];
ry(-0.8727183275683225) q[3];
cx q[1],q[3];
ry(-2.985403421137993) q[1];
ry(1.330014958838359) q[3];
cx q[1],q[3];
ry(3.062954206994464) q[0];
ry(2.990241300323602) q[1];
cx q[0],q[1];
ry(-0.39234756676985244) q[0];
ry(-1.8947651786213404) q[1];
cx q[0],q[1];
ry(-0.19973031556381127) q[2];
ry(-2.199318803628334) q[3];
cx q[2],q[3];
ry(1.8213350960675854) q[2];
ry(-0.3594955739837751) q[3];
cx q[2],q[3];
ry(1.7605949147274738) q[0];
ry(-1.885591818755081) q[2];
cx q[0],q[2];
ry(1.912251517250132) q[0];
ry(-0.5469891459473013) q[2];
cx q[0],q[2];
ry(0.37492085445090184) q[1];
ry(1.1851425788634753) q[3];
cx q[1],q[3];
ry(2.4731612303421304) q[1];
ry(1.4493241890593787) q[3];
cx q[1],q[3];
ry(0.32045671637614187) q[0];
ry(0.3720472005408669) q[1];
cx q[0],q[1];
ry(2.9821817641183928) q[0];
ry(0.47596900965941735) q[1];
cx q[0],q[1];
ry(1.2229875467829088) q[2];
ry(0.5156018418876087) q[3];
cx q[2],q[3];
ry(-1.9129522602733653) q[2];
ry(-0.34676226514356934) q[3];
cx q[2],q[3];
ry(2.4722552190726215) q[0];
ry(-1.1039900435626495) q[2];
cx q[0],q[2];
ry(-1.0330486451462837) q[0];
ry(1.094544282518714) q[2];
cx q[0],q[2];
ry(1.492451499694542) q[1];
ry(-0.2985167049635582) q[3];
cx q[1],q[3];
ry(-2.7735748021780533) q[1];
ry(1.0213600970158072) q[3];
cx q[1],q[3];
ry(-2.7923073752054504) q[0];
ry(-1.0909503026985536) q[1];
cx q[0],q[1];
ry(1.4460854758607509) q[0];
ry(2.455825584468057) q[1];
cx q[0],q[1];
ry(1.798552335617754) q[2];
ry(2.14648022486598) q[3];
cx q[2],q[3];
ry(-0.5062056587293692) q[2];
ry(-0.2813799058647818) q[3];
cx q[2],q[3];
ry(-1.284441075582687) q[0];
ry(-2.878724556952456) q[2];
cx q[0],q[2];
ry(2.4511276449979054) q[0];
ry(1.0730587479767753) q[2];
cx q[0],q[2];
ry(0.5989307987566502) q[1];
ry(-1.903779021798038) q[3];
cx q[1],q[3];
ry(2.7177883351057854) q[1];
ry(-0.6604435079299048) q[3];
cx q[1],q[3];
ry(-0.1563818304277017) q[0];
ry(0.02205545112679186) q[1];
cx q[0],q[1];
ry(-1.5655868499386418) q[0];
ry(0.9544290357735249) q[1];
cx q[0],q[1];
ry(-0.8356549114167384) q[2];
ry(-0.14894023691946823) q[3];
cx q[2],q[3];
ry(2.671061463054685) q[2];
ry(3.061599575285834) q[3];
cx q[2],q[3];
ry(-2.58839738603814) q[0];
ry(1.8186738784360632) q[2];
cx q[0],q[2];
ry(2.05965639678385) q[0];
ry(-0.7113494680463202) q[2];
cx q[0],q[2];
ry(-2.5661533890729324) q[1];
ry(-1.885042794631835) q[3];
cx q[1],q[3];
ry(0.07342557577458156) q[1];
ry(-1.594023642550985) q[3];
cx q[1],q[3];
ry(0.48841592364829906) q[0];
ry(2.6564787067255295) q[1];
cx q[0],q[1];
ry(2.767684099781212) q[0];
ry(2.8636297863847893) q[1];
cx q[0],q[1];
ry(-1.2895670489523485) q[2];
ry(-0.5629474864122548) q[3];
cx q[2],q[3];
ry(-2.9403416028294) q[2];
ry(-1.5331321341289241) q[3];
cx q[2],q[3];
ry(-2.723461833673189) q[0];
ry(-1.1356297435562355) q[2];
cx q[0],q[2];
ry(-3.1327513982112816) q[0];
ry(-0.31522885100271036) q[2];
cx q[0],q[2];
ry(0.8579199551049029) q[1];
ry(-2.0012949997928446) q[3];
cx q[1],q[3];
ry(1.9000498849829182) q[1];
ry(0.2970968640806416) q[3];
cx q[1],q[3];
ry(-2.2728707135936803) q[0];
ry(-2.0084028266705594) q[1];
cx q[0],q[1];
ry(0.8761505084099257) q[0];
ry(0.8129659913798442) q[1];
cx q[0],q[1];
ry(-1.1687206880710788) q[2];
ry(2.060288662914772) q[3];
cx q[2],q[3];
ry(0.5929126459639343) q[2];
ry(2.624519111447434) q[3];
cx q[2],q[3];
ry(1.2598215621858457) q[0];
ry(1.4289013997886872) q[2];
cx q[0],q[2];
ry(1.5697062626406053) q[0];
ry(-2.418512688443217) q[2];
cx q[0],q[2];
ry(-2.8402087438789048) q[1];
ry(-0.4275669561893149) q[3];
cx q[1],q[3];
ry(-0.5612454694510847) q[1];
ry(-1.4527127200576615) q[3];
cx q[1],q[3];
ry(-1.250072071537819) q[0];
ry(2.6633609150443194) q[1];
cx q[0],q[1];
ry(1.3791261624601576) q[0];
ry(0.8484030161027567) q[1];
cx q[0],q[1];
ry(-2.821387170035404) q[2];
ry(1.5992510872648467) q[3];
cx q[2],q[3];
ry(1.6069906949762316) q[2];
ry(1.3215694885239784) q[3];
cx q[2],q[3];
ry(1.7426395486653712) q[0];
ry(-2.6248582214944265) q[2];
cx q[0],q[2];
ry(1.940161930500735) q[0];
ry(-0.5207303140581585) q[2];
cx q[0],q[2];
ry(2.646762930579021) q[1];
ry(-0.2291316169463026) q[3];
cx q[1],q[3];
ry(1.576503842284473) q[1];
ry(2.3426522302105903) q[3];
cx q[1],q[3];
ry(1.4839366209556688) q[0];
ry(-1.6595786033492983) q[1];
cx q[0],q[1];
ry(-0.9932222366668765) q[0];
ry(0.26501927102845535) q[1];
cx q[0],q[1];
ry(-2.2958466429534536) q[2];
ry(-0.5791713195113335) q[3];
cx q[2],q[3];
ry(-1.4934174135130354) q[2];
ry(-2.598338010592981) q[3];
cx q[2],q[3];
ry(0.24543187792706375) q[0];
ry(-2.256467511539692) q[2];
cx q[0],q[2];
ry(-1.9277040103907632) q[0];
ry(0.02893138960744784) q[2];
cx q[0],q[2];
ry(-1.8897903366116324) q[1];
ry(-0.11949483820114773) q[3];
cx q[1],q[3];
ry(-0.9109244679118338) q[1];
ry(-2.2242642617602177) q[3];
cx q[1],q[3];
ry(-2.2227316190149935) q[0];
ry(-2.713962626159873) q[1];
cx q[0],q[1];
ry(-2.5964063601866934) q[0];
ry(0.8434162936279826) q[1];
cx q[0],q[1];
ry(2.863332101753336) q[2];
ry(2.1280080256660217) q[3];
cx q[2],q[3];
ry(-2.5594117279129445) q[2];
ry(2.2952233093253063) q[3];
cx q[2],q[3];
ry(-2.966932046424571) q[0];
ry(-0.2909995708078741) q[2];
cx q[0],q[2];
ry(3.0551773363636996) q[0];
ry(0.21457317093214545) q[2];
cx q[0],q[2];
ry(0.3601963680413364) q[1];
ry(-0.10562765241239802) q[3];
cx q[1],q[3];
ry(0.4896733577883792) q[1];
ry(-2.2463591585619858) q[3];
cx q[1],q[3];
ry(-0.8486026657709829) q[0];
ry(-0.9498952965456446) q[1];
cx q[0],q[1];
ry(-0.9142258153393482) q[0];
ry(-1.0830060974348914) q[1];
cx q[0],q[1];
ry(-1.464086390561193) q[2];
ry(-1.3198070795732173) q[3];
cx q[2],q[3];
ry(0.5335830830414459) q[2];
ry(1.4236486497366032) q[3];
cx q[2],q[3];
ry(-0.7070450590700961) q[0];
ry(2.413133197634496) q[2];
cx q[0],q[2];
ry(2.1028542765025047) q[0];
ry(-2.4406692093871802) q[2];
cx q[0],q[2];
ry(0.5582725105764119) q[1];
ry(-1.020589570692601) q[3];
cx q[1],q[3];
ry(-0.7891008169626399) q[1];
ry(-0.042851537847204746) q[3];
cx q[1],q[3];
ry(1.554116345763181) q[0];
ry(1.5171390176904695) q[1];
cx q[0],q[1];
ry(1.1486388563678158) q[0];
ry(-0.017636738930600537) q[1];
cx q[0],q[1];
ry(-2.4692265324144054) q[2];
ry(-1.1946562353526984) q[3];
cx q[2],q[3];
ry(2.4015576780310144) q[2];
ry(0.2864663757565352) q[3];
cx q[2],q[3];
ry(-2.803400544522337) q[0];
ry(1.030527411972792) q[2];
cx q[0],q[2];
ry(-0.4358686858042331) q[0];
ry(0.6015917854885656) q[2];
cx q[0],q[2];
ry(-0.06285772281522384) q[1];
ry(-2.8156752054338376) q[3];
cx q[1],q[3];
ry(1.2161058399247846) q[1];
ry(-1.8149813397404362) q[3];
cx q[1],q[3];
ry(1.0157259734319615) q[0];
ry(0.8397321166363797) q[1];
cx q[0],q[1];
ry(1.66021680696398) q[0];
ry(0.19396667463186504) q[1];
cx q[0],q[1];
ry(-1.769051509081345) q[2];
ry(-1.5816078762853623) q[3];
cx q[2],q[3];
ry(0.8107953142834772) q[2];
ry(-3.0595754926653442) q[3];
cx q[2],q[3];
ry(-0.011239949735988915) q[0];
ry(-2.384430291099906) q[2];
cx q[0],q[2];
ry(0.1207636222188471) q[0];
ry(0.7008074938996431) q[2];
cx q[0],q[2];
ry(-0.11731026390967686) q[1];
ry(0.09647449842920515) q[3];
cx q[1],q[3];
ry(-1.395206033118943) q[1];
ry(-0.9938418122995887) q[3];
cx q[1],q[3];
ry(2.267536186724449) q[0];
ry(-1.4561035503851096) q[1];
cx q[0],q[1];
ry(2.468770996618037) q[0];
ry(-1.9197765819029051) q[1];
cx q[0],q[1];
ry(0.024357868197747903) q[2];
ry(0.3412668535767145) q[3];
cx q[2],q[3];
ry(2.8682404915119344) q[2];
ry(-3.118337658602026) q[3];
cx q[2],q[3];
ry(-2.680027544134926) q[0];
ry(-1.026281846881167) q[2];
cx q[0],q[2];
ry(-1.9357448635671657) q[0];
ry(-0.9657418379791722) q[2];
cx q[0],q[2];
ry(1.5840270695652037) q[1];
ry(-2.3456033574023607) q[3];
cx q[1],q[3];
ry(0.6873564657790148) q[1];
ry(-1.9593493288165) q[3];
cx q[1],q[3];
ry(-0.20775759386649906) q[0];
ry(1.9120240920794496) q[1];
cx q[0],q[1];
ry(1.3928042988677527) q[0];
ry(-2.255466944523582) q[1];
cx q[0],q[1];
ry(-0.45866092291324895) q[2];
ry(-2.068121929579986) q[3];
cx q[2],q[3];
ry(-0.12486396080204543) q[2];
ry(-2.3296985743661085) q[3];
cx q[2],q[3];
ry(-1.7470761713958476) q[0];
ry(-0.5675725297091564) q[2];
cx q[0],q[2];
ry(-0.31874417112323017) q[0];
ry(2.5486481427044154) q[2];
cx q[0],q[2];
ry(1.2159903656120594) q[1];
ry(1.0995219188462868) q[3];
cx q[1],q[3];
ry(-2.4208870657928765) q[1];
ry(-0.21353463368120076) q[3];
cx q[1],q[3];
ry(1.9959880016610079) q[0];
ry(1.680129704041366) q[1];
cx q[0],q[1];
ry(-0.9878870327550205) q[0];
ry(-0.266591024873021) q[1];
cx q[0],q[1];
ry(-2.81781784874172) q[2];
ry(-0.020714547816854934) q[3];
cx q[2],q[3];
ry(2.312557134981143) q[2];
ry(-0.08639869698666304) q[3];
cx q[2],q[3];
ry(0.35751134172712) q[0];
ry(1.8362482003176233) q[2];
cx q[0],q[2];
ry(2.9207603065575904) q[0];
ry(2.4468715463382793) q[2];
cx q[0],q[2];
ry(1.433116744561925) q[1];
ry(-1.3422367524555396) q[3];
cx q[1],q[3];
ry(-1.8857008055536566) q[1];
ry(3.1103548548777846) q[3];
cx q[1],q[3];
ry(-0.17139044869732792) q[0];
ry(-2.1063990127332364) q[1];
cx q[0],q[1];
ry(-0.7406953246800289) q[0];
ry(-2.1899187960848794) q[1];
cx q[0],q[1];
ry(2.1859960728512715) q[2];
ry(-0.3711228906347075) q[3];
cx q[2],q[3];
ry(1.4799485757308692) q[2];
ry(-0.10061426057144394) q[3];
cx q[2],q[3];
ry(1.8026771714410659) q[0];
ry(0.5403047785217168) q[2];
cx q[0],q[2];
ry(-0.3455806752874153) q[0];
ry(2.881707516299474) q[2];
cx q[0],q[2];
ry(0.924870510564647) q[1];
ry(0.5977764957013996) q[3];
cx q[1],q[3];
ry(-0.8123098126307423) q[1];
ry(-0.519167908337677) q[3];
cx q[1],q[3];
ry(-0.4486430192491624) q[0];
ry(-0.03661594329225881) q[1];
cx q[0],q[1];
ry(0.6515389436396236) q[0];
ry(2.7900930790386127) q[1];
cx q[0],q[1];
ry(1.4117336187717484) q[2];
ry(3.0920497774339077) q[3];
cx q[2],q[3];
ry(-2.952072656367655) q[2];
ry(1.715832632508806) q[3];
cx q[2],q[3];
ry(-0.5911689023896766) q[0];
ry(-0.8894826074590955) q[2];
cx q[0],q[2];
ry(2.6766317002084214) q[0];
ry(2.168648611384483) q[2];
cx q[0],q[2];
ry(-2.0636816743041835) q[1];
ry(-1.8228067029152104) q[3];
cx q[1],q[3];
ry(-0.845028624028119) q[1];
ry(2.588111295810207) q[3];
cx q[1],q[3];
ry(1.293250214735143) q[0];
ry(1.6193283208494522) q[1];
ry(-2.131379224860204) q[2];
ry(-1.7778169573721023) q[3];