OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.30566159823188865) q[0];
ry(-2.9130303580938377) q[1];
cx q[0],q[1];
ry(2.6943125951504863) q[0];
ry(-1.9722182909650423) q[1];
cx q[0],q[1];
ry(0.9251602594574857) q[2];
ry(-2.899633317594353) q[3];
cx q[2],q[3];
ry(-3.0101522820678586) q[2];
ry(0.9295432395359288) q[3];
cx q[2],q[3];
ry(-0.8537249530518212) q[4];
ry(-0.575697665403261) q[5];
cx q[4],q[5];
ry(-0.24851860385136515) q[4];
ry(0.3213096963848177) q[5];
cx q[4],q[5];
ry(1.9578316894304337) q[6];
ry(-1.3074889306048156) q[7];
cx q[6],q[7];
ry(-1.8512039559616005) q[6];
ry(-2.100650795858782) q[7];
cx q[6],q[7];
ry(1.1295811259702855) q[8];
ry(-1.9481124073392708) q[9];
cx q[8],q[9];
ry(-0.04560369388580913) q[8];
ry(-0.4908622770156476) q[9];
cx q[8],q[9];
ry(0.5715658601611562) q[10];
ry(-0.9573665708623251) q[11];
cx q[10],q[11];
ry(-0.10619970143002623) q[10];
ry(-2.827292134245159) q[11];
cx q[10],q[11];
ry(2.859656698137356) q[1];
ry(-0.4430550466200236) q[2];
cx q[1],q[2];
ry(0.5212164652202933) q[1];
ry(1.2203056665480778) q[2];
cx q[1],q[2];
ry(1.1848806017680908) q[3];
ry(2.352368750259289) q[4];
cx q[3],q[4];
ry(1.691075117146199) q[3];
ry(-1.7937267367738967) q[4];
cx q[3],q[4];
ry(0.3883282345638124) q[5];
ry(-0.3698515965872794) q[6];
cx q[5],q[6];
ry(-3.044179282346596) q[5];
ry(-3.0274251948664554) q[6];
cx q[5],q[6];
ry(0.28536125351397723) q[7];
ry(-1.0438268637788772) q[8];
cx q[7],q[8];
ry(2.0340748429802504) q[7];
ry(-0.8091463766107421) q[8];
cx q[7],q[8];
ry(-0.24772478036212736) q[9];
ry(0.759879789617063) q[10];
cx q[9],q[10];
ry(-0.38889796594774234) q[9];
ry(-1.8369099259807293) q[10];
cx q[9],q[10];
ry(0.31375141896017134) q[0];
ry(-0.510214800961597) q[1];
cx q[0],q[1];
ry(-3.019281320544774) q[0];
ry(0.2217911706457132) q[1];
cx q[0],q[1];
ry(-0.9534301563997671) q[2];
ry(3.1031987354460697) q[3];
cx q[2],q[3];
ry(-1.043482732160662) q[2];
ry(1.1257689840363225) q[3];
cx q[2],q[3];
ry(1.8464170961398496) q[4];
ry(1.499377713090949) q[5];
cx q[4],q[5];
ry(0.06566939937575786) q[4];
ry(2.936386527274896) q[5];
cx q[4],q[5];
ry(1.1071346269322841) q[6];
ry(-2.4045054492994358) q[7];
cx q[6],q[7];
ry(2.1517956105001863) q[6];
ry(-2.7141034755016795) q[7];
cx q[6],q[7];
ry(-1.6622532661995648) q[8];
ry(0.3235634257352008) q[9];
cx q[8],q[9];
ry(1.329584528057432) q[8];
ry(0.9080190948422361) q[9];
cx q[8],q[9];
ry(-3.0071893178923297) q[10];
ry(2.383210627198674) q[11];
cx q[10],q[11];
ry(2.0344231333527087) q[10];
ry(-3.0885893412812333) q[11];
cx q[10],q[11];
ry(0.3464146528653846) q[1];
ry(2.8760959217056614) q[2];
cx q[1],q[2];
ry(1.5384477233720935) q[1];
ry(2.208509144530434) q[2];
cx q[1],q[2];
ry(2.6522860028409494) q[3];
ry(-0.6798193882528248) q[4];
cx q[3],q[4];
ry(-2.9845007602091904) q[3];
ry(1.6349482862705302) q[4];
cx q[3],q[4];
ry(-0.029247691556078072) q[5];
ry(-2.9289580710018694) q[6];
cx q[5],q[6];
ry(-1.742549225022234) q[5];
ry(0.026237485231400903) q[6];
cx q[5],q[6];
ry(-0.16866469206515244) q[7];
ry(2.9276619762883387) q[8];
cx q[7],q[8];
ry(1.7658973646970049) q[7];
ry(-0.48158181913842535) q[8];
cx q[7],q[8];
ry(0.29773035740841447) q[9];
ry(2.493537509521735) q[10];
cx q[9],q[10];
ry(1.3727574864311978) q[9];
ry(-1.0053483644030177) q[10];
cx q[9],q[10];
ry(-2.321128085801783) q[0];
ry(1.264653207912886) q[1];
cx q[0],q[1];
ry(1.466789485976637) q[0];
ry(0.8729143079857985) q[1];
cx q[0],q[1];
ry(-1.5726140969906788) q[2];
ry(-2.6212431897001256) q[3];
cx q[2],q[3];
ry(-0.11190947681430242) q[2];
ry(0.35357371004255445) q[3];
cx q[2],q[3];
ry(2.526762079233062) q[4];
ry(1.624057667067447) q[5];
cx q[4],q[5];
ry(-3.118530784916208) q[4];
ry(1.5544498288144444) q[5];
cx q[4],q[5];
ry(0.4586080364443577) q[6];
ry(0.21527937207483472) q[7];
cx q[6],q[7];
ry(-0.056873866061379286) q[6];
ry(-0.10102108129296729) q[7];
cx q[6],q[7];
ry(1.2396310017126613) q[8];
ry(0.9542803680939986) q[9];
cx q[8],q[9];
ry(-0.5823013798107857) q[8];
ry(1.760774891873754) q[9];
cx q[8],q[9];
ry(2.592897206616425) q[10];
ry(0.6668511590112143) q[11];
cx q[10],q[11];
ry(0.2943766687404692) q[10];
ry(-0.13115421759646023) q[11];
cx q[10],q[11];
ry(-1.5884874845538501) q[1];
ry(2.6476842529499933) q[2];
cx q[1],q[2];
ry(0.4588812435484666) q[1];
ry(-2.1214011052856785) q[2];
cx q[1],q[2];
ry(1.0812223912929424) q[3];
ry(1.5147350693690944) q[4];
cx q[3],q[4];
ry(-0.47293584589175897) q[3];
ry(-0.1586074563089148) q[4];
cx q[3],q[4];
ry(1.523084503554683) q[5];
ry(-1.9489315581953937) q[6];
cx q[5],q[6];
ry(-2.0391377485212603) q[5];
ry(-1.5770043785400876) q[6];
cx q[5],q[6];
ry(-1.9465487281683327) q[7];
ry(2.9438362165004195) q[8];
cx q[7],q[8];
ry(0.08145525270672567) q[7];
ry(-2.394897547254657) q[8];
cx q[7],q[8];
ry(0.04039959990641635) q[9];
ry(2.1166016404255172) q[10];
cx q[9],q[10];
ry(0.8162206268956504) q[9];
ry(0.19022146137036255) q[10];
cx q[9],q[10];
ry(2.702817073844386) q[0];
ry(-2.572918051029101) q[1];
cx q[0],q[1];
ry(-0.41947130999677995) q[0];
ry(0.5732668753619131) q[1];
cx q[0],q[1];
ry(-3.015968042056274) q[2];
ry(2.2476467932199826) q[3];
cx q[2],q[3];
ry(0.7438830783225986) q[2];
ry(2.9777216728500333) q[3];
cx q[2],q[3];
ry(-0.7313668795474468) q[4];
ry(-1.6058799957598093) q[5];
cx q[4],q[5];
ry(-0.1936615729654204) q[4];
ry(-0.061148523684892354) q[5];
cx q[4],q[5];
ry(-2.3519444447111906) q[6];
ry(-2.3468687752673922) q[7];
cx q[6],q[7];
ry(-0.09211712646229173) q[6];
ry(-2.8860477592613596) q[7];
cx q[6],q[7];
ry(2.0719811710186824) q[8];
ry(-2.674959927597067) q[9];
cx q[8],q[9];
ry(-0.9221137709185436) q[8];
ry(2.211781829889099) q[9];
cx q[8],q[9];
ry(-0.041097640511675415) q[10];
ry(2.249523599234645) q[11];
cx q[10],q[11];
ry(-2.7998324185662535) q[10];
ry(3.0239655567366044) q[11];
cx q[10],q[11];
ry(-2.8788870924170085) q[1];
ry(1.3635943670128259) q[2];
cx q[1],q[2];
ry(-1.7245412014131505) q[1];
ry(-2.950642500992692) q[2];
cx q[1],q[2];
ry(-2.208930322821619) q[3];
ry(-0.2836044305323533) q[4];
cx q[3],q[4];
ry(-3.130691057395803) q[3];
ry(0.32757849672277506) q[4];
cx q[3],q[4];
ry(-0.17084705218756824) q[5];
ry(2.227304534970209) q[6];
cx q[5],q[6];
ry(2.8339782525579174) q[5];
ry(-0.02178297591905373) q[6];
cx q[5],q[6];
ry(1.5641064544887775) q[7];
ry(2.4980526593046015) q[8];
cx q[7],q[8];
ry(-0.9151671498314683) q[7];
ry(0.9694717422247993) q[8];
cx q[7],q[8];
ry(3.100377225871706) q[9];
ry(-1.4237210555774737) q[10];
cx q[9],q[10];
ry(2.9291018289101913) q[9];
ry(-1.6600621760187737) q[10];
cx q[9],q[10];
ry(2.3338716224495926) q[0];
ry(-0.5891279178422225) q[1];
cx q[0],q[1];
ry(-0.9443852012101379) q[0];
ry(-3.123564654404862) q[1];
cx q[0],q[1];
ry(2.4537010455317527) q[2];
ry(-1.2664672818262188) q[3];
cx q[2],q[3];
ry(-0.9703407885038658) q[2];
ry(-0.06320457067866414) q[3];
cx q[2],q[3];
ry(0.5864046716855315) q[4];
ry(1.3537108907434074) q[5];
cx q[4],q[5];
ry(-0.09981843856660788) q[4];
ry(0.48554323328186655) q[5];
cx q[4],q[5];
ry(0.9169080606134175) q[6];
ry(-1.3430298819580564) q[7];
cx q[6],q[7];
ry(-0.12160475658880521) q[6];
ry(-2.8634790227588054) q[7];
cx q[6],q[7];
ry(2.8676726424127787) q[8];
ry(-1.6113837261708281) q[9];
cx q[8],q[9];
ry(1.6903416195182033) q[8];
ry(-1.4573315951133186) q[9];
cx q[8],q[9];
ry(-0.5556094574792221) q[10];
ry(-0.5152693104324322) q[11];
cx q[10],q[11];
ry(-1.5368779310871759) q[10];
ry(-0.01857185569890607) q[11];
cx q[10],q[11];
ry(2.427860849031579) q[1];
ry(1.9870541983445886) q[2];
cx q[1],q[2];
ry(2.6624179682977003) q[1];
ry(2.43471436317025) q[2];
cx q[1],q[2];
ry(-0.8337664391620017) q[3];
ry(1.5402197620974682) q[4];
cx q[3],q[4];
ry(0.3139291413760015) q[3];
ry(-0.5807828339827923) q[4];
cx q[3],q[4];
ry(0.842873809018454) q[5];
ry(-1.9079240237065713) q[6];
cx q[5],q[6];
ry(-2.9920673022695428) q[5];
ry(-0.004765314615326163) q[6];
cx q[5],q[6];
ry(2.646522002770876) q[7];
ry(1.990038965352129) q[8];
cx q[7],q[8];
ry(2.5133149130488297) q[7];
ry(-2.30591659291792) q[8];
cx q[7],q[8];
ry(-1.7375143704973484) q[9];
ry(1.9921431983959546) q[10];
cx q[9],q[10];
ry(-1.3809896154006407) q[9];
ry(1.8472865839534425) q[10];
cx q[9],q[10];
ry(-3.029107058502241) q[0];
ry(3.0646338044587766) q[1];
cx q[0],q[1];
ry(1.9512086241244833) q[0];
ry(-0.24426580387817196) q[1];
cx q[0],q[1];
ry(-0.6971528936292752) q[2];
ry(-1.3165907360684466) q[3];
cx q[2],q[3];
ry(0.010360055192326634) q[2];
ry(3.1080605701519812) q[3];
cx q[2],q[3];
ry(-1.6845880220455518) q[4];
ry(-1.6663876825238964) q[5];
cx q[4],q[5];
ry(-0.2589447905439295) q[4];
ry(-0.06389948607157248) q[5];
cx q[4],q[5];
ry(1.2445665917586946) q[6];
ry(-1.9543118728659081) q[7];
cx q[6],q[7];
ry(0.8184177838529694) q[6];
ry(-0.932664713902084) q[7];
cx q[6],q[7];
ry(-2.9206663696879485) q[8];
ry(-0.2860418939760905) q[9];
cx q[8],q[9];
ry(-1.4589853422393313) q[8];
ry(-1.7970111495451966) q[9];
cx q[8],q[9];
ry(-1.595037861818124) q[10];
ry(2.1001909222315467) q[11];
cx q[10],q[11];
ry(0.05692574225141023) q[10];
ry(-0.08233681649121838) q[11];
cx q[10],q[11];
ry(2.2272936752262407) q[1];
ry(0.20030474092672185) q[2];
cx q[1],q[2];
ry(0.013941120471918643) q[1];
ry(2.598556917554419) q[2];
cx q[1],q[2];
ry(0.6071724112824768) q[3];
ry(2.5196659684292215) q[4];
cx q[3],q[4];
ry(-2.01474927916542) q[3];
ry(-1.488926913994252) q[4];
cx q[3],q[4];
ry(-1.5536752759835881) q[5];
ry(-0.9681764736242098) q[6];
cx q[5],q[6];
ry(-3.104448668460851) q[5];
ry(-0.02705302272963649) q[6];
cx q[5],q[6];
ry(1.8260744971637664) q[7];
ry(0.7501105505600689) q[8];
cx q[7],q[8];
ry(-1.6423740755478375) q[7];
ry(-0.7711381752837181) q[8];
cx q[7],q[8];
ry(-0.8304182062238559) q[9];
ry(-2.322892314213475) q[10];
cx q[9],q[10];
ry(-1.4662703187820911) q[9];
ry(-1.2765342921778036) q[10];
cx q[9],q[10];
ry(1.603756320902427) q[0];
ry(0.4017074251298668) q[1];
cx q[0],q[1];
ry(-2.35544548747197) q[0];
ry(0.17156828374696076) q[1];
cx q[0],q[1];
ry(2.059691869411835) q[2];
ry(1.7156752985998094) q[3];
cx q[2],q[3];
ry(0.005360958037859298) q[2];
ry(3.125058348245237) q[3];
cx q[2],q[3];
ry(-2.825201621809994) q[4];
ry(1.3949655590610153) q[5];
cx q[4],q[5];
ry(-3.140706063942201) q[4];
ry(-1.9781873127301397) q[5];
cx q[4],q[5];
ry(-1.6606067965350677) q[6];
ry(1.2287375773190352) q[7];
cx q[6],q[7];
ry(-1.0587656255005182) q[6];
ry(0.9693984019040992) q[7];
cx q[6],q[7];
ry(1.3926175628155066) q[8];
ry(0.9739616905058259) q[9];
cx q[8],q[9];
ry(0.40279684462904086) q[8];
ry(0.6834094882934741) q[9];
cx q[8],q[9];
ry(-0.6453659825161191) q[10];
ry(1.2241836417145706) q[11];
cx q[10],q[11];
ry(0.8678110523871428) q[10];
ry(-3.100916289119091) q[11];
cx q[10],q[11];
ry(0.2797001769658829) q[1];
ry(-1.3998486413016364) q[2];
cx q[1],q[2];
ry(2.765046313422856) q[1];
ry(-1.3382159395155613) q[2];
cx q[1],q[2];
ry(0.17454857688187886) q[3];
ry(1.5934653326822494) q[4];
cx q[3],q[4];
ry(-2.3784317208960957) q[3];
ry(-3.14112877874205) q[4];
cx q[3],q[4];
ry(-2.666088725234926) q[5];
ry(0.7870867641549592) q[6];
cx q[5],q[6];
ry(-0.12276544533309243) q[5];
ry(-0.0006174352511703063) q[6];
cx q[5],q[6];
ry(-1.2427191815315644) q[7];
ry(0.9228179085058077) q[8];
cx q[7],q[8];
ry(0.43733352154533944) q[7];
ry(2.690352625269557) q[8];
cx q[7],q[8];
ry(-1.0741935906111344) q[9];
ry(-1.1325133432652157) q[10];
cx q[9],q[10];
ry(-1.269777021430028) q[9];
ry(-2.9522042590521043) q[10];
cx q[9],q[10];
ry(2.2924255261694904) q[0];
ry(2.002889634189085) q[1];
cx q[0],q[1];
ry(1.35692522561773) q[0];
ry(2.4988013862739216) q[1];
cx q[0],q[1];
ry(-0.71176661120142) q[2];
ry(1.584663101394822) q[3];
cx q[2],q[3];
ry(3.1345667687516907) q[2];
ry(1.903561406109359) q[3];
cx q[2],q[3];
ry(-3.0619105752463223) q[4];
ry(-0.7005464031061743) q[5];
cx q[4],q[5];
ry(-1.5012794309243063) q[4];
ry(-1.4930801073044897) q[5];
cx q[4],q[5];
ry(1.7008639752035632) q[6];
ry(-2.9814755443962127) q[7];
cx q[6],q[7];
ry(0.1294453341028112) q[6];
ry(2.7937967429009083) q[7];
cx q[6],q[7];
ry(-0.5639686400699354) q[8];
ry(-1.630766928167067) q[9];
cx q[8],q[9];
ry(1.2969713972329944) q[8];
ry(-0.42484967217250436) q[9];
cx q[8],q[9];
ry(-1.9248198543020774) q[10];
ry(1.6101353217333074) q[11];
cx q[10],q[11];
ry(-1.7414613616021741) q[10];
ry(-0.5966841310397868) q[11];
cx q[10],q[11];
ry(3.140108510060917) q[1];
ry(1.5963258819701303) q[2];
cx q[1],q[2];
ry(-1.7344609160502067) q[1];
ry(0.0435459118804134) q[2];
cx q[1],q[2];
ry(-0.8122598342776874) q[3];
ry(-0.41202001656643716) q[4];
cx q[3],q[4];
ry(2.4736812124868437) q[3];
ry(2.4829765024832553) q[4];
cx q[3],q[4];
ry(-1.5646395124384043) q[5];
ry(-2.8039762286211025) q[6];
cx q[5],q[6];
ry(-1.5983070092535119) q[5];
ry(-2.9151781304618605) q[6];
cx q[5],q[6];
ry(0.30135445880148387) q[7];
ry(-1.2755260645810154) q[8];
cx q[7],q[8];
ry(-1.8335985799654928) q[7];
ry(1.9578275034564414) q[8];
cx q[7],q[8];
ry(1.3450551767447525) q[9];
ry(0.18384276867168658) q[10];
cx q[9],q[10];
ry(-0.05263440898642813) q[9];
ry(1.5631027563946753) q[10];
cx q[9],q[10];
ry(-1.8277565310663884) q[0];
ry(2.615911838617757) q[1];
cx q[0],q[1];
ry(0.08815722836161743) q[0];
ry(-0.8917938921530899) q[1];
cx q[0],q[1];
ry(-1.5795935152526797) q[2];
ry(0.9656512766883801) q[3];
cx q[2],q[3];
ry(-1.5905631291126225) q[2];
ry(-2.521379948310603) q[3];
cx q[2],q[3];
ry(1.6165786472925454) q[4];
ry(-0.6436336314979645) q[5];
cx q[4],q[5];
ry(1.575480761666712) q[4];
ry(-1.4291198685070519) q[5];
cx q[4],q[5];
ry(0.8139687588704255) q[6];
ry(-2.2776172967320605) q[7];
cx q[6],q[7];
ry(-2.224472897271285) q[6];
ry(-1.531962437743422) q[7];
cx q[6],q[7];
ry(1.893412013043918) q[8];
ry(-2.6612916112566327) q[9];
cx q[8],q[9];
ry(0.5451366706175937) q[8];
ry(0.013741149412670708) q[9];
cx q[8],q[9];
ry(-0.1888855330276762) q[10];
ry(0.15029474963000666) q[11];
cx q[10],q[11];
ry(-2.905819654566289) q[10];
ry(2.3501362472265472) q[11];
cx q[10],q[11];
ry(-1.1184978910980854) q[1];
ry(1.5469148353501891) q[2];
cx q[1],q[2];
ry(-1.5321230966687551) q[1];
ry(1.2480254146768022) q[2];
cx q[1],q[2];
ry(-3.0969509208665555) q[3];
ry(1.2656656506993802) q[4];
cx q[3],q[4];
ry(0.00023344964995963992) q[3];
ry(3.1397129409431868) q[4];
cx q[3],q[4];
ry(0.021566857400437737) q[5];
ry(2.364966629818779) q[6];
cx q[5],q[6];
ry(3.1398657446182567) q[5];
ry(0.060935189677914905) q[6];
cx q[5],q[6];
ry(-1.8471025569556847) q[7];
ry(-1.998434942832267) q[8];
cx q[7],q[8];
ry(-3.141254057041214) q[7];
ry(-3.1395261734976034) q[8];
cx q[7],q[8];
ry(0.8405725885223116) q[9];
ry(-2.171486065703473) q[10];
cx q[9],q[10];
ry(3.016390777182043) q[9];
ry(1.3929500169664009) q[10];
cx q[9],q[10];
ry(0.6127586533489531) q[0];
ry(3.04579342108387) q[1];
cx q[0],q[1];
ry(2.868596299643657) q[0];
ry(1.0621717805383528) q[1];
cx q[0],q[1];
ry(2.8310827940919348) q[2];
ry(-1.893979756171639) q[3];
cx q[2],q[3];
ry(-1.799114223217775) q[2];
ry(-2.906062799474455) q[3];
cx q[2],q[3];
ry(-2.851007339061242) q[4];
ry(-3.1338377517904337) q[5];
cx q[4],q[5];
ry(-1.6081070564852178) q[4];
ry(1.5880396924763391) q[5];
cx q[4],q[5];
ry(-2.379136923751604) q[6];
ry(1.1795071878404646) q[7];
cx q[6],q[7];
ry(1.9840852164065268) q[6];
ry(0.7163956150051733) q[7];
cx q[6],q[7];
ry(-1.7870221604141676) q[8];
ry(-1.953590419735213) q[9];
cx q[8],q[9];
ry(-0.9293625522806819) q[8];
ry(-2.1289050336999544) q[9];
cx q[8],q[9];
ry(0.07250047830015399) q[10];
ry(-0.37236588036160617) q[11];
cx q[10],q[11];
ry(0.05108071671979435) q[10];
ry(-1.4729420206284187) q[11];
cx q[10],q[11];
ry(-0.9711229218216564) q[1];
ry(-0.2617967303996205) q[2];
cx q[1],q[2];
ry(-0.31948687659809044) q[1];
ry(3.134950937839392) q[2];
cx q[1],q[2];
ry(-2.756565053942049) q[3];
ry(-1.449909084591356) q[4];
cx q[3],q[4];
ry(-0.0020571506009223494) q[3];
ry(-3.1336888740605726) q[4];
cx q[3],q[4];
ry(-0.6693295588373935) q[5];
ry(-2.9579594993332337) q[6];
cx q[5],q[6];
ry(3.1104014591611295) q[5];
ry(0.14395534457429807) q[6];
cx q[5],q[6];
ry(-0.33762657561207376) q[7];
ry(-2.3548223108849533) q[8];
cx q[7],q[8];
ry(-3.126665497801422) q[7];
ry(-3.141132682509314) q[8];
cx q[7],q[8];
ry(-1.9844648409630943) q[9];
ry(-3.1256912213389834) q[10];
cx q[9],q[10];
ry(2.209297305875488) q[9];
ry(-2.6690087965671934) q[10];
cx q[9],q[10];
ry(0.16835768245904156) q[0];
ry(-0.943641377431991) q[1];
cx q[0],q[1];
ry(-0.19800810246348455) q[0];
ry(2.216149987479791) q[1];
cx q[0],q[1];
ry(1.5267137889358748) q[2];
ry(-1.8275351485732871) q[3];
cx q[2],q[3];
ry(2.8206147552496343) q[2];
ry(-1.5617053397843481) q[3];
cx q[2],q[3];
ry(-1.706112956788782) q[4];
ry(0.9762173147237867) q[5];
cx q[4],q[5];
ry(-1.6193693348025713) q[4];
ry(-1.5105900239575512) q[5];
cx q[4],q[5];
ry(2.3233610154505597) q[6];
ry(-2.240306437349017) q[7];
cx q[6],q[7];
ry(-1.575781424314508) q[6];
ry(0.09323746316025651) q[7];
cx q[6],q[7];
ry(-2.35354994868858) q[8];
ry(1.5779174853718996) q[9];
cx q[8],q[9];
ry(-1.4249640012313736) q[8];
ry(-1.6612795034252574) q[9];
cx q[8],q[9];
ry(-0.6213910221529466) q[10];
ry(0.3901333186793998) q[11];
cx q[10],q[11];
ry(3.1225556184163588) q[10];
ry(-3.037412482207854) q[11];
cx q[10],q[11];
ry(0.17787383574447213) q[1];
ry(-1.4088339602235773) q[2];
cx q[1],q[2];
ry(-0.887403085336835) q[1];
ry(0.06247218203525584) q[2];
cx q[1],q[2];
ry(-2.1363584185602207) q[3];
ry(-1.1598042619059272) q[4];
cx q[3],q[4];
ry(0.0001765856287210357) q[3];
ry(3.139867481714709) q[4];
cx q[3],q[4];
ry(0.9254289043940097) q[5];
ry(-0.6013316861691023) q[6];
cx q[5],q[6];
ry(-3.057857938394924) q[5];
ry(0.0063153828922109054) q[6];
cx q[5],q[6];
ry(-1.831354888726115) q[7];
ry(-0.46580609440675325) q[8];
cx q[7],q[8];
ry(-0.5658912957469182) q[7];
ry(0.7836345134290365) q[8];
cx q[7],q[8];
ry(1.569472068652418) q[9];
ry(-1.7952031497952943) q[10];
cx q[9],q[10];
ry(0.6756735413916554) q[9];
ry(-0.6797466291752858) q[10];
cx q[9],q[10];
ry(2.9779836262262838) q[0];
ry(0.2705854505151297) q[1];
cx q[0],q[1];
ry(-1.4331229396467746) q[0];
ry(3.0806559539533214) q[1];
cx q[0],q[1];
ry(2.7905507890435644) q[2];
ry(-2.206000116454743) q[3];
cx q[2],q[3];
ry(-1.4828732165778018) q[2];
ry(-0.19998588448865956) q[3];
cx q[2],q[3];
ry(-0.295480127571841) q[4];
ry(-2.514436135298784) q[5];
cx q[4],q[5];
ry(-1.9098860791020487) q[4];
ry(-2.8572087914571718) q[5];
cx q[4],q[5];
ry(-0.039586745641513606) q[6];
ry(1.9657432235270198) q[7];
cx q[6],q[7];
ry(0.0036641830141788713) q[6];
ry(-3.0221604818946575) q[7];
cx q[6],q[7];
ry(1.1385904901365667) q[8];
ry(-2.5269692942659865) q[9];
cx q[8],q[9];
ry(-3.139261622026642) q[8];
ry(-3.139270896678293) q[9];
cx q[8],q[9];
ry(1.0119461449033162) q[10];
ry(-1.7011796597485322) q[11];
cx q[10],q[11];
ry(2.7340286644980547) q[10];
ry(-1.4375471596765825) q[11];
cx q[10],q[11];
ry(2.106505026356457) q[1];
ry(-1.2151385905725718) q[2];
cx q[1],q[2];
ry(-2.998464597880421) q[1];
ry(-0.014179680426668284) q[2];
cx q[1],q[2];
ry(2.366750317632862) q[3];
ry(-1.6321815452753479) q[4];
cx q[3],q[4];
ry(-3.1263839329448233) q[3];
ry(3.139653986565286) q[4];
cx q[3],q[4];
ry(-0.00867299888967299) q[5];
ry(1.6174773764090369) q[6];
cx q[5],q[6];
ry(-1.5715330579216047) q[5];
ry(-1.5844536764950419) q[6];
cx q[5],q[6];
ry(-0.15923476914496243) q[7];
ry(-0.8242802650784142) q[8];
cx q[7],q[8];
ry(-1.817324658562666) q[7];
ry(-1.3472087963339912) q[8];
cx q[7],q[8];
ry(-2.515412675396566) q[9];
ry(2.7453876556918257) q[10];
cx q[9],q[10];
ry(-0.4715631604398771) q[9];
ry(1.5567642911620263) q[10];
cx q[9],q[10];
ry(1.596428676703291) q[0];
ry(-0.5643252754561121) q[1];
cx q[0],q[1];
ry(1.1006887132002887) q[0];
ry(0.4056961658979568) q[1];
cx q[0],q[1];
ry(0.7321413531068277) q[2];
ry(1.5667977501827575) q[3];
cx q[2],q[3];
ry(-0.0729619168808759) q[2];
ry(2.962022603311124) q[3];
cx q[2],q[3];
ry(-1.744073558555134) q[4];
ry(-2.2030764773612113) q[5];
cx q[4],q[5];
ry(-0.0005207132493119104) q[4];
ry(-1.5864612331707415) q[5];
cx q[4],q[5];
ry(0.0236410295528664) q[6];
ry(-2.6652399576639207) q[7];
cx q[6],q[7];
ry(0.00508869231486837) q[6];
ry(-3.1236461506692454) q[7];
cx q[6],q[7];
ry(0.5203491581070183) q[8];
ry(1.5757968930007156) q[9];
cx q[8],q[9];
ry(1.5704634859485216) q[8];
ry(2.9041612463083335) q[9];
cx q[8],q[9];
ry(-2.204127814160171) q[10];
ry(3.1286741241441316) q[11];
cx q[10],q[11];
ry(1.7519573174996756) q[10];
ry(1.4773572024802586) q[11];
cx q[10],q[11];
ry(0.2267241287730959) q[1];
ry(1.051010762937325) q[2];
cx q[1],q[2];
ry(0.04528611469121739) q[1];
ry(3.0228999513468855) q[2];
cx q[1],q[2];
ry(-2.4479162246632926) q[3];
ry(1.5989623227788385) q[4];
cx q[3],q[4];
ry(-1.5428250189018187) q[3];
ry(-3.101914903950761) q[4];
cx q[3],q[4];
ry(2.6936208586745565) q[5];
ry(1.6905990112329543) q[6];
cx q[5],q[6];
ry(-3.0884901257134842) q[5];
ry(-0.026876324035020834) q[6];
cx q[5],q[6];
ry(1.8296324232713221) q[7];
ry(-1.7404892841364392) q[8];
cx q[7],q[8];
ry(-3.1411335856986558) q[7];
ry(-1.8850403033190644) q[8];
cx q[7],q[8];
ry(1.5582968054568118) q[9];
ry(-1.5089423005548344) q[10];
cx q[9],q[10];
ry(-1.5685677492213346) q[9];
ry(-0.046991192129171466) q[10];
cx q[9],q[10];
ry(-0.4505209884887116) q[0];
ry(-0.03776449982005942) q[1];
cx q[0],q[1];
ry(-0.1292211517772941) q[0];
ry(-3.0285087038567347) q[1];
cx q[0],q[1];
ry(-0.041059152092528375) q[2];
ry(-1.6036242052337215) q[3];
cx q[2],q[3];
ry(-0.7428784490201753) q[2];
ry(0.16092719916486775) q[3];
cx q[2],q[3];
ry(-3.065445379862748) q[4];
ry(-2.0273971065381846) q[5];
cx q[4],q[5];
ry(-2.5206159199903304) q[4];
ry(0.0022493458576455794) q[5];
cx q[4],q[5];
ry(-3.018714515880558) q[6];
ry(0.037981200289369355) q[7];
cx q[6],q[7];
ry(0.17468999417049672) q[6];
ry(-1.5713256611332262) q[7];
cx q[6],q[7];
ry(2.764152312661352) q[8];
ry(-1.5581719193326895) q[9];
cx q[8],q[9];
ry(-1.3784945439227396) q[8];
ry(2.791794447638613) q[9];
cx q[8],q[9];
ry(-1.0969941623277497) q[10];
ry(3.1409159549304815) q[11];
cx q[10],q[11];
ry(-1.5710134645311669) q[10];
ry(-3.1411293702890126) q[11];
cx q[10],q[11];
ry(0.021236950105728663) q[1];
ry(1.4283316201336458) q[2];
cx q[1],q[2];
ry(3.128413954141897) q[1];
ry(1.4686471553340477) q[2];
cx q[1],q[2];
ry(1.5705531704999585) q[3];
ry(3.06557859247036) q[4];
cx q[3],q[4];
ry(1.684682455729134) q[3];
ry(1.5275477622475815) q[4];
cx q[3],q[4];
ry(-1.5710610279584116) q[5];
ry(-1.5477447690304444) q[6];
cx q[5],q[6];
ry(-3.13622063557717) q[5];
ry(-1.5713452571573339) q[6];
cx q[5],q[6];
ry(-0.005902816860225428) q[7];
ry(1.3525132006138683) q[8];
cx q[7],q[8];
ry(1.5286472725224582) q[7];
ry(-3.1411980513019944) q[8];
cx q[7],q[8];
ry(3.0316151365285027) q[9];
ry(2.0437740118032792) q[10];
cx q[9],q[10];
ry(0.7117140678865379) q[9];
ry(0.0003469213231342703) q[10];
cx q[9],q[10];
ry(3.1276469542174676) q[0];
ry(1.5843616676790748) q[1];
ry(-3.1003196013992502) q[2];
ry(1.5716412898748973) q[3];
ry(3.1407304413903767) q[4];
ry(-1.5707495757628829) q[5];
ry(0.017197301292363854) q[6];
ry(-3.1077042673817408) q[7];
ry(-0.00015559539800432418) q[8];
ry(-3.0315356372439446) q[9];
ry(-3.140430093713627) q[10];
ry(1.5698698990006577) q[11];