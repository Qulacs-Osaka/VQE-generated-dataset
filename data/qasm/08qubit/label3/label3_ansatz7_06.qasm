OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.7946463406050113) q[0];
ry(2.6076260368786772) q[1];
cx q[0],q[1];
ry(-0.217168774213888) q[0];
ry(2.7083494689214715) q[1];
cx q[0],q[1];
ry(-1.9749853241509376) q[0];
ry(1.510894969313551) q[2];
cx q[0],q[2];
ry(-0.4342333778294571) q[0];
ry(-2.3412760548370235) q[2];
cx q[0],q[2];
ry(-0.8368489123517833) q[0];
ry(1.7148937697072375) q[3];
cx q[0],q[3];
ry(0.853550167471135) q[0];
ry(-2.4873158538357574) q[3];
cx q[0],q[3];
ry(0.8783950101478082) q[0];
ry(1.1736972188873276) q[4];
cx q[0],q[4];
ry(-1.7915152046858238) q[0];
ry(-2.2552831561745315) q[4];
cx q[0],q[4];
ry(-3.0331779984992338) q[0];
ry(-0.2829097849158695) q[5];
cx q[0],q[5];
ry(-2.0817803293892054) q[0];
ry(-1.621486487645038) q[5];
cx q[0],q[5];
ry(-2.2058258876748096) q[0];
ry(1.2295957442591716) q[6];
cx q[0],q[6];
ry(-0.6973883818041182) q[0];
ry(2.399802322601866) q[6];
cx q[0],q[6];
ry(0.04082627974173114) q[0];
ry(-1.0268603818097848) q[7];
cx q[0],q[7];
ry(2.964541646982136) q[0];
ry(-1.903107945526969) q[7];
cx q[0],q[7];
ry(-2.6792693052770407) q[1];
ry(-0.7676623158837641) q[2];
cx q[1],q[2];
ry(2.8333355528583564) q[1];
ry(0.9914793791840424) q[2];
cx q[1],q[2];
ry(-0.7283076429558266) q[1];
ry(-1.495688483715564) q[3];
cx q[1],q[3];
ry(-2.0830193421164447) q[1];
ry(-0.12187053114757253) q[3];
cx q[1],q[3];
ry(0.3024417872512108) q[1];
ry(-2.2407912358722157) q[4];
cx q[1],q[4];
ry(-2.3936009826804985) q[1];
ry(0.15641454799468055) q[4];
cx q[1],q[4];
ry(-1.0248497359924433) q[1];
ry(-3.1131249808834967) q[5];
cx q[1],q[5];
ry(2.566273681081331) q[1];
ry(-0.7906327066041045) q[5];
cx q[1],q[5];
ry(-1.1230480461592194) q[1];
ry(-3.004390081837309) q[6];
cx q[1],q[6];
ry(2.9422448181277137) q[1];
ry(-0.5203512601335779) q[6];
cx q[1],q[6];
ry(0.021202554163727218) q[1];
ry(2.3205622657683613) q[7];
cx q[1],q[7];
ry(1.5182445451191606) q[1];
ry(2.2729489275787076) q[7];
cx q[1],q[7];
ry(-1.0545193123746441) q[2];
ry(-0.7984481458445829) q[3];
cx q[2],q[3];
ry(-0.348405659584742) q[2];
ry(-0.6320154280166621) q[3];
cx q[2],q[3];
ry(-2.729433154579824) q[2];
ry(-2.121602250117792) q[4];
cx q[2],q[4];
ry(-0.591466357106146) q[2];
ry(-2.1896173934885734) q[4];
cx q[2],q[4];
ry(1.3707810434589687) q[2];
ry(-0.4072012811393964) q[5];
cx q[2],q[5];
ry(-1.1710760965034674) q[2];
ry(0.21617454895677032) q[5];
cx q[2],q[5];
ry(2.8435347904691834) q[2];
ry(-1.9428133817523205) q[6];
cx q[2],q[6];
ry(-0.9262554653784819) q[2];
ry(2.936683257702216) q[6];
cx q[2],q[6];
ry(-0.35601115544827255) q[2];
ry(0.5269902235488302) q[7];
cx q[2],q[7];
ry(-1.5733988543224706) q[2];
ry(-0.661837193354142) q[7];
cx q[2],q[7];
ry(-0.6426851485726406) q[3];
ry(2.070723854570127) q[4];
cx q[3],q[4];
ry(-1.3566472442481345) q[3];
ry(0.9757236893075409) q[4];
cx q[3],q[4];
ry(2.218857208110161) q[3];
ry(2.929084474904265) q[5];
cx q[3],q[5];
ry(-0.8028648722237719) q[3];
ry(-2.55861823603123) q[5];
cx q[3],q[5];
ry(-0.29335524069900254) q[3];
ry(-0.7470960359655523) q[6];
cx q[3],q[6];
ry(0.29400428372703846) q[3];
ry(2.559645049589972) q[6];
cx q[3],q[6];
ry(0.8202348461665175) q[3];
ry(2.5961246944511998) q[7];
cx q[3],q[7];
ry(-1.004130619328654) q[3];
ry(-2.94976973691917) q[7];
cx q[3],q[7];
ry(-2.3288915184178256) q[4];
ry(-0.22776734312761351) q[5];
cx q[4],q[5];
ry(-1.972176902876062) q[4];
ry(2.9556749644232627) q[5];
cx q[4],q[5];
ry(-0.6505388149531323) q[4];
ry(3.0307576686209425) q[6];
cx q[4],q[6];
ry(-2.3313240293649975) q[4];
ry(-2.411967531452761) q[6];
cx q[4],q[6];
ry(2.187991550897756) q[4];
ry(3.0281969890656675) q[7];
cx q[4],q[7];
ry(0.4302152303437866) q[4];
ry(-2.082914643007262) q[7];
cx q[4],q[7];
ry(0.2399144429733617) q[5];
ry(3.0231806194329356) q[6];
cx q[5],q[6];
ry(-1.555935674662149) q[5];
ry(-2.105793239360057) q[6];
cx q[5],q[6];
ry(-2.4696691978076735) q[5];
ry(1.2750149812765024) q[7];
cx q[5],q[7];
ry(-1.4988153316428807) q[5];
ry(0.06528620436708012) q[7];
cx q[5],q[7];
ry(2.1087374096927602) q[6];
ry(-0.31361684147174507) q[7];
cx q[6],q[7];
ry(-3.0758697447731294) q[6];
ry(-0.8803152683712466) q[7];
cx q[6],q[7];
ry(0.6123696514109671) q[0];
ry(1.5879677502008747) q[1];
cx q[0],q[1];
ry(0.8694562235279832) q[0];
ry(-1.5256613798998935) q[1];
cx q[0],q[1];
ry(3.0364168232474427) q[0];
ry(-0.7196749944086495) q[2];
cx q[0],q[2];
ry(1.107809607657873) q[0];
ry(-2.1519808066232056) q[2];
cx q[0],q[2];
ry(2.4407169681934358) q[0];
ry(-1.5730809633140215) q[3];
cx q[0],q[3];
ry(3.069069683822973) q[0];
ry(-1.5886971799855676) q[3];
cx q[0],q[3];
ry(-0.9474954144184142) q[0];
ry(0.5551062441062058) q[4];
cx q[0],q[4];
ry(1.194093379654806) q[0];
ry(-2.894752118535769) q[4];
cx q[0],q[4];
ry(-1.8957349110274988) q[0];
ry(0.1633154978783005) q[5];
cx q[0],q[5];
ry(1.7762630305519087) q[0];
ry(2.6630285586633957) q[5];
cx q[0],q[5];
ry(1.4438608748085484) q[0];
ry(0.11277810483158034) q[6];
cx q[0],q[6];
ry(-0.6326876790460298) q[0];
ry(1.969677729800666) q[6];
cx q[0],q[6];
ry(-0.08298380839456865) q[0];
ry(-0.4156573403470914) q[7];
cx q[0],q[7];
ry(0.6703507072804803) q[0];
ry(0.15689182844891678) q[7];
cx q[0],q[7];
ry(-0.04811045520413428) q[1];
ry(-2.7089360004704663) q[2];
cx q[1],q[2];
ry(-0.5537810421761797) q[1];
ry(-3.138557013945268) q[2];
cx q[1],q[2];
ry(-2.6218304715397207) q[1];
ry(-0.9762511722716916) q[3];
cx q[1],q[3];
ry(0.5869068829496888) q[1];
ry(-1.8243982929308222) q[3];
cx q[1],q[3];
ry(-0.6708262536776046) q[1];
ry(-2.7460127587913) q[4];
cx q[1],q[4];
ry(-0.5437049590064271) q[1];
ry(-1.3873354028301557) q[4];
cx q[1],q[4];
ry(0.08330194554184889) q[1];
ry(0.22533775932600866) q[5];
cx q[1],q[5];
ry(-3.057803248536424) q[1];
ry(2.00928794814831) q[5];
cx q[1],q[5];
ry(0.841673515621956) q[1];
ry(1.7699750394864715) q[6];
cx q[1],q[6];
ry(-1.8039936660720004) q[1];
ry(3.132392673995322) q[6];
cx q[1],q[6];
ry(0.5911703478930415) q[1];
ry(-1.2409958359575892) q[7];
cx q[1],q[7];
ry(-2.5539675034032414) q[1];
ry(-2.634618760894021) q[7];
cx q[1],q[7];
ry(-2.3701685208324603) q[2];
ry(3.1192182247222466) q[3];
cx q[2],q[3];
ry(1.0412206410190779) q[2];
ry(0.9619684848462462) q[3];
cx q[2],q[3];
ry(0.8861319893944524) q[2];
ry(1.2226649484647725) q[4];
cx q[2],q[4];
ry(-1.9052183493531354) q[2];
ry(-2.645014754571531) q[4];
cx q[2],q[4];
ry(-2.525511324659241) q[2];
ry(0.9446576182670414) q[5];
cx q[2],q[5];
ry(-0.2676952847114827) q[2];
ry(0.6649515137060767) q[5];
cx q[2],q[5];
ry(-2.4353846876371867) q[2];
ry(2.6433136638054955) q[6];
cx q[2],q[6];
ry(-2.1602309877485935) q[2];
ry(-2.0359940943142925) q[6];
cx q[2],q[6];
ry(1.147530211021782) q[2];
ry(-1.6762362033270357) q[7];
cx q[2],q[7];
ry(-1.8594360209485465) q[2];
ry(-1.2015145453576555) q[7];
cx q[2],q[7];
ry(1.0703617671757888) q[3];
ry(1.013571067349059) q[4];
cx q[3],q[4];
ry(-2.6390088538025434) q[3];
ry(2.158665149348824) q[4];
cx q[3],q[4];
ry(-2.875481622194285) q[3];
ry(2.5720467016468413) q[5];
cx q[3],q[5];
ry(2.2305831341911944) q[3];
ry(-2.277938260296727) q[5];
cx q[3],q[5];
ry(-1.3494462467931951) q[3];
ry(-2.8144021912769284) q[6];
cx q[3],q[6];
ry(2.8514625442345958) q[3];
ry(-2.8992457497084625) q[6];
cx q[3],q[6];
ry(2.7428322653466375) q[3];
ry(2.122152218511175) q[7];
cx q[3],q[7];
ry(-1.5800908036960548) q[3];
ry(-2.2125931087274475) q[7];
cx q[3],q[7];
ry(-0.8815094345251041) q[4];
ry(1.2891610054130886) q[5];
cx q[4],q[5];
ry(-1.5464398732926496) q[4];
ry(2.4945820097793767) q[5];
cx q[4],q[5];
ry(1.5507720101249785) q[4];
ry(0.5355170523333856) q[6];
cx q[4],q[6];
ry(-2.8129814289696036) q[4];
ry(-0.8106456103192228) q[6];
cx q[4],q[6];
ry(0.46315337995630845) q[4];
ry(1.064385108343079) q[7];
cx q[4],q[7];
ry(0.5021398101138363) q[4];
ry(2.8991789733091378) q[7];
cx q[4],q[7];
ry(0.8275789113328302) q[5];
ry(0.5316813693913403) q[6];
cx q[5],q[6];
ry(-1.5674492902227986) q[5];
ry(0.6626558026743088) q[6];
cx q[5],q[6];
ry(1.1188724123119773) q[5];
ry(0.11176894776575708) q[7];
cx q[5],q[7];
ry(2.1058726750691523) q[5];
ry(-1.5789892990971137) q[7];
cx q[5],q[7];
ry(0.32052033856291084) q[6];
ry(1.1676877289887686) q[7];
cx q[6],q[7];
ry(0.24389499872332987) q[6];
ry(2.4340891463854675) q[7];
cx q[6],q[7];
ry(-2.0935659421857395) q[0];
ry(-3.1286090651569824) q[1];
cx q[0],q[1];
ry(-0.5160224257793625) q[0];
ry(-2.090397655362839) q[1];
cx q[0],q[1];
ry(-1.0493155164548307) q[0];
ry(1.793447678348958) q[2];
cx q[0],q[2];
ry(-3.007647117183185) q[0];
ry(-1.553782843055006) q[2];
cx q[0],q[2];
ry(1.5958373089402267) q[0];
ry(2.6748730305231843) q[3];
cx q[0],q[3];
ry(-2.7957943033235804) q[0];
ry(-0.767342330400302) q[3];
cx q[0],q[3];
ry(2.987355807193901) q[0];
ry(1.193170507124998) q[4];
cx q[0],q[4];
ry(2.3660583174296734) q[0];
ry(-0.9467005080236124) q[4];
cx q[0],q[4];
ry(0.18303852299917756) q[0];
ry(-1.7967536034597649) q[5];
cx q[0],q[5];
ry(2.8109283403443106) q[0];
ry(1.5836294666436606) q[5];
cx q[0],q[5];
ry(2.3025453287742508) q[0];
ry(-0.520401663813459) q[6];
cx q[0],q[6];
ry(0.31873131761753964) q[0];
ry(-1.7380247283626422) q[6];
cx q[0],q[6];
ry(2.4431869185238493) q[0];
ry(-2.4190647026957963) q[7];
cx q[0],q[7];
ry(-2.971316850969681) q[0];
ry(2.6012125471340726) q[7];
cx q[0],q[7];
ry(-1.5176339392672353) q[1];
ry(-1.462703566600259) q[2];
cx q[1],q[2];
ry(-2.1758939844921237) q[1];
ry(1.063262291719325) q[2];
cx q[1],q[2];
ry(-2.249183398828293) q[1];
ry(1.7169144708309827) q[3];
cx q[1],q[3];
ry(0.9221075426928316) q[1];
ry(2.743092840528107) q[3];
cx q[1],q[3];
ry(-0.5308098918156752) q[1];
ry(1.469140586076234) q[4];
cx q[1],q[4];
ry(-2.1072777806844174) q[1];
ry(-0.4148107109765078) q[4];
cx q[1],q[4];
ry(-2.360465283312084) q[1];
ry(2.7396252850656237) q[5];
cx q[1],q[5];
ry(1.935220077987618) q[1];
ry(-0.17096201322664406) q[5];
cx q[1],q[5];
ry(-0.24204784653166467) q[1];
ry(-0.3018124719810351) q[6];
cx q[1],q[6];
ry(2.0976399753095785) q[1];
ry(-2.1211691836794677) q[6];
cx q[1],q[6];
ry(-2.2661733348958277) q[1];
ry(2.3865616009722577) q[7];
cx q[1],q[7];
ry(0.38644083228648) q[1];
ry(-1.157417213846716) q[7];
cx q[1],q[7];
ry(-1.4627287486305205) q[2];
ry(-2.2427646865472735) q[3];
cx q[2],q[3];
ry(1.2974376728082404) q[2];
ry(1.448607001740407) q[3];
cx q[2],q[3];
ry(0.1465753870615858) q[2];
ry(-2.6980856427637816) q[4];
cx q[2],q[4];
ry(-3.0296670623081314) q[2];
ry(-1.157083880351018) q[4];
cx q[2],q[4];
ry(-0.8110878070239744) q[2];
ry(1.92088595153057) q[5];
cx q[2],q[5];
ry(3.0824098987070045) q[2];
ry(2.4000740457020666) q[5];
cx q[2],q[5];
ry(2.810401700206145) q[2];
ry(2.0831649798120226) q[6];
cx q[2],q[6];
ry(0.9816561878644655) q[2];
ry(1.4131546546955558) q[6];
cx q[2],q[6];
ry(2.499972131885591) q[2];
ry(-1.0381660468789704) q[7];
cx q[2],q[7];
ry(-2.2266074974287586) q[2];
ry(-0.4860348986041154) q[7];
cx q[2],q[7];
ry(-2.1714516615384074) q[3];
ry(-1.8963227313874942) q[4];
cx q[3],q[4];
ry(1.1372735168750092) q[3];
ry(-2.940307606154024) q[4];
cx q[3],q[4];
ry(-1.499005345991347) q[3];
ry(-2.1148819417199727) q[5];
cx q[3],q[5];
ry(0.005136768456878826) q[3];
ry(-0.5683637055314246) q[5];
cx q[3],q[5];
ry(1.773113672703241) q[3];
ry(-1.1121323851237657) q[6];
cx q[3],q[6];
ry(-2.2641112351861277) q[3];
ry(1.205974113367619) q[6];
cx q[3],q[6];
ry(-0.8169047077089003) q[3];
ry(2.556860160996218) q[7];
cx q[3],q[7];
ry(1.5293356286129915) q[3];
ry(-0.6172728179431459) q[7];
cx q[3],q[7];
ry(-1.21947016502576) q[4];
ry(1.2235924167348375) q[5];
cx q[4],q[5];
ry(0.45010164856555057) q[4];
ry(3.092019102789003) q[5];
cx q[4],q[5];
ry(2.014217808625376) q[4];
ry(-2.5784401695862744) q[6];
cx q[4],q[6];
ry(2.70323612759392) q[4];
ry(-0.9362390407369618) q[6];
cx q[4],q[6];
ry(2.1567856606333864) q[4];
ry(-3.1352014507400745) q[7];
cx q[4],q[7];
ry(-2.02000847673202) q[4];
ry(-0.9147393004618495) q[7];
cx q[4],q[7];
ry(-0.4439208063502927) q[5];
ry(2.8664789288588537) q[6];
cx q[5],q[6];
ry(2.15415662750312) q[5];
ry(0.41315869049547516) q[6];
cx q[5],q[6];
ry(0.797609050192098) q[5];
ry(-1.7670153161441355) q[7];
cx q[5],q[7];
ry(1.345182714754298) q[5];
ry(1.983088269762363) q[7];
cx q[5],q[7];
ry(1.8541794992304652) q[6];
ry(1.971428625276551) q[7];
cx q[6],q[7];
ry(-2.15774375279413) q[6];
ry(0.7994673069211506) q[7];
cx q[6],q[7];
ry(-1.9389895898685703) q[0];
ry(-1.4644144320330017) q[1];
cx q[0],q[1];
ry(-2.054322647310206) q[0];
ry(0.7635048744089872) q[1];
cx q[0],q[1];
ry(-3.100892346542798) q[0];
ry(-2.1191214915273964) q[2];
cx q[0],q[2];
ry(2.0120033583251526) q[0];
ry(1.3817454725855605) q[2];
cx q[0],q[2];
ry(1.3469991313658505) q[0];
ry(2.8104043339491946) q[3];
cx q[0],q[3];
ry(-1.5638557821556707) q[0];
ry(-2.7425805558804113) q[3];
cx q[0],q[3];
ry(-1.5303248842562243) q[0];
ry(-1.359224040121073) q[4];
cx q[0],q[4];
ry(0.11202858789826167) q[0];
ry(-2.8509586754386147) q[4];
cx q[0],q[4];
ry(-0.5860837767033287) q[0];
ry(-1.8392910448068518) q[5];
cx q[0],q[5];
ry(0.8204578707691614) q[0];
ry(2.73215172851719) q[5];
cx q[0],q[5];
ry(0.5185084985349493) q[0];
ry(-2.065535864682654) q[6];
cx q[0],q[6];
ry(0.852557816939304) q[0];
ry(2.2592577789965462) q[6];
cx q[0],q[6];
ry(-2.498361018412195) q[0];
ry(2.600341871642975) q[7];
cx q[0],q[7];
ry(1.9550092252399196) q[0];
ry(1.1022497531194198) q[7];
cx q[0],q[7];
ry(0.5927407485854638) q[1];
ry(-0.2583568668696641) q[2];
cx q[1],q[2];
ry(-2.8558957432223147) q[1];
ry(2.743173253830406) q[2];
cx q[1],q[2];
ry(-1.689802170756364) q[1];
ry(0.9424528170403068) q[3];
cx q[1],q[3];
ry(-0.7469985562054662) q[1];
ry(1.270036316033294) q[3];
cx q[1],q[3];
ry(-2.491124578095605) q[1];
ry(0.8845942102009259) q[4];
cx q[1],q[4];
ry(-3.124644250047624) q[1];
ry(1.4366781895830618) q[4];
cx q[1],q[4];
ry(-2.3041309674658326) q[1];
ry(2.330321864678502) q[5];
cx q[1],q[5];
ry(1.7799207051220238) q[1];
ry(2.6344665595756207) q[5];
cx q[1],q[5];
ry(-1.5293923007140497) q[1];
ry(-0.9378371287564238) q[6];
cx q[1],q[6];
ry(-2.135111292687794) q[1];
ry(-0.22853533779928714) q[6];
cx q[1],q[6];
ry(-1.1409451768768804) q[1];
ry(1.5134491297655617) q[7];
cx q[1],q[7];
ry(-0.00201949085203168) q[1];
ry(1.983865119287737) q[7];
cx q[1],q[7];
ry(1.62136876435131) q[2];
ry(1.9258540095912409) q[3];
cx q[2],q[3];
ry(-0.6009781595919593) q[2];
ry(0.7274670387737355) q[3];
cx q[2],q[3];
ry(0.5242333700588065) q[2];
ry(2.9198725893737025) q[4];
cx q[2],q[4];
ry(-1.974077847526638) q[2];
ry(-2.818597361154699) q[4];
cx q[2],q[4];
ry(-2.4222237177628325) q[2];
ry(-1.2839113743493107) q[5];
cx q[2],q[5];
ry(-2.84634088738205) q[2];
ry(1.279993826528096) q[5];
cx q[2],q[5];
ry(0.2377368521872755) q[2];
ry(1.4090885342560668) q[6];
cx q[2],q[6];
ry(2.4625356282537587) q[2];
ry(2.323019747366885) q[6];
cx q[2],q[6];
ry(-2.2380364396680372) q[2];
ry(3.024695574628719) q[7];
cx q[2],q[7];
ry(0.5879973933410452) q[2];
ry(1.6801381311403896) q[7];
cx q[2],q[7];
ry(0.0744323699501595) q[3];
ry(-1.7608804290767035) q[4];
cx q[3],q[4];
ry(2.7588250512737167) q[3];
ry(-1.0259815710319735) q[4];
cx q[3],q[4];
ry(-1.2983467036187113) q[3];
ry(2.458640281440012) q[5];
cx q[3],q[5];
ry(1.1960029196545792) q[3];
ry(2.3696409339935207) q[5];
cx q[3],q[5];
ry(-2.1258913153632597) q[3];
ry(-2.6904323027243313) q[6];
cx q[3],q[6];
ry(1.529952960986971) q[3];
ry(1.7557224356706786) q[6];
cx q[3],q[6];
ry(0.6510035258643613) q[3];
ry(-1.2058963754601937) q[7];
cx q[3],q[7];
ry(1.4714502358524502) q[3];
ry(2.092969701606382) q[7];
cx q[3],q[7];
ry(0.17191048997836322) q[4];
ry(-0.8024920730329042) q[5];
cx q[4],q[5];
ry(-0.1744481247438672) q[4];
ry(0.5967395167946357) q[5];
cx q[4],q[5];
ry(-1.5033173617291278) q[4];
ry(-2.420223790861243) q[6];
cx q[4],q[6];
ry(0.859669323681599) q[4];
ry(-2.0197417258875108) q[6];
cx q[4],q[6];
ry(2.90312018226083) q[4];
ry(0.5720443784303288) q[7];
cx q[4],q[7];
ry(1.9888961751452063) q[4];
ry(2.2674807470268235) q[7];
cx q[4],q[7];
ry(-2.125015449788766) q[5];
ry(1.9493456094043555) q[6];
cx q[5],q[6];
ry(3.09513297637236) q[5];
ry(-1.8715130370621598) q[6];
cx q[5],q[6];
ry(-2.0631988549561378) q[5];
ry(-1.887442515879478) q[7];
cx q[5],q[7];
ry(-1.6785209769759186) q[5];
ry(2.1704370718898023) q[7];
cx q[5],q[7];
ry(2.2670103904752468) q[6];
ry(0.4262616088873754) q[7];
cx q[6],q[7];
ry(-0.14855368168913594) q[6];
ry(-0.5243129179142203) q[7];
cx q[6],q[7];
ry(-2.896847398923512) q[0];
ry(3.1235188958552023) q[1];
cx q[0],q[1];
ry(-0.28867540071836206) q[0];
ry(0.8244343918517799) q[1];
cx q[0],q[1];
ry(2.408860841401158) q[0];
ry(-0.6148557174259056) q[2];
cx q[0],q[2];
ry(0.24663815700946243) q[0];
ry(-1.8456012695284754) q[2];
cx q[0],q[2];
ry(2.0898587461885496) q[0];
ry(1.5616311457547198) q[3];
cx q[0],q[3];
ry(-2.994534010162245) q[0];
ry(2.942475009994573) q[3];
cx q[0],q[3];
ry(2.2464175499497774) q[0];
ry(-1.8537382757632255) q[4];
cx q[0],q[4];
ry(-1.119247287472179) q[0];
ry(2.059709838606777) q[4];
cx q[0],q[4];
ry(2.205812167576995) q[0];
ry(-0.9777962908069977) q[5];
cx q[0],q[5];
ry(-2.3452975639251123) q[0];
ry(-1.04561168224709) q[5];
cx q[0],q[5];
ry(-0.65482314845425) q[0];
ry(0.6672502409203442) q[6];
cx q[0],q[6];
ry(-3.0860789712176846) q[0];
ry(-2.411225252290275) q[6];
cx q[0],q[6];
ry(0.4682982828014284) q[0];
ry(1.544740347165092) q[7];
cx q[0],q[7];
ry(1.5034653625824532) q[0];
ry(-2.592691501497967) q[7];
cx q[0],q[7];
ry(-0.7104219476023008) q[1];
ry(-1.8706084107824825) q[2];
cx q[1],q[2];
ry(1.6703168301662688) q[1];
ry(-2.9974238384673635) q[2];
cx q[1],q[2];
ry(-0.5230329296622169) q[1];
ry(2.2205608034227113) q[3];
cx q[1],q[3];
ry(-0.3980721831515819) q[1];
ry(1.2351654988082839) q[3];
cx q[1],q[3];
ry(0.7757117328179559) q[1];
ry(-2.786278705603637) q[4];
cx q[1],q[4];
ry(-2.2542870482645325) q[1];
ry(1.7993267913645707) q[4];
cx q[1],q[4];
ry(2.542813109835559) q[1];
ry(1.4691457232256777) q[5];
cx q[1],q[5];
ry(-0.6619516499226092) q[1];
ry(-2.123667248240407) q[5];
cx q[1],q[5];
ry(2.469649103995463) q[1];
ry(1.8865162193113862) q[6];
cx q[1],q[6];
ry(0.9978232225062728) q[1];
ry(1.8908754975606845) q[6];
cx q[1],q[6];
ry(1.8030208199004738) q[1];
ry(-2.7089124934287425) q[7];
cx q[1],q[7];
ry(1.5840569230662158) q[1];
ry(-1.8052707983679759) q[7];
cx q[1],q[7];
ry(-2.1694483903682675) q[2];
ry(-1.2915642313068316) q[3];
cx q[2],q[3];
ry(1.8590626382519917) q[2];
ry(-2.7347832641689407) q[3];
cx q[2],q[3];
ry(0.13723495084539541) q[2];
ry(0.3015101268776434) q[4];
cx q[2],q[4];
ry(3.0718817376931233) q[2];
ry(-0.7291121705405219) q[4];
cx q[2],q[4];
ry(-1.5836021030424199) q[2];
ry(-1.3746836553896369) q[5];
cx q[2],q[5];
ry(-2.3880973459575006) q[2];
ry(2.562145937468342) q[5];
cx q[2],q[5];
ry(-2.173297874154107) q[2];
ry(-1.846100389558187) q[6];
cx q[2],q[6];
ry(1.8983939354347426) q[2];
ry(0.0880205474415261) q[6];
cx q[2],q[6];
ry(2.59791730719577) q[2];
ry(1.3609717102103263) q[7];
cx q[2],q[7];
ry(-0.9178709288027153) q[2];
ry(0.7031832027447855) q[7];
cx q[2],q[7];
ry(-1.3891845052602232) q[3];
ry(2.2667833601425933) q[4];
cx q[3],q[4];
ry(1.595395388136962) q[3];
ry(0.9167441859774331) q[4];
cx q[3],q[4];
ry(-2.1788368266099134) q[3];
ry(0.9188952563703409) q[5];
cx q[3],q[5];
ry(1.780000346873428) q[3];
ry(1.3420310642033284) q[5];
cx q[3],q[5];
ry(2.8080794589532494) q[3];
ry(-0.8470933073254497) q[6];
cx q[3],q[6];
ry(-2.824971450649149) q[3];
ry(-2.5268150990359675) q[6];
cx q[3],q[6];
ry(-0.022165903369224083) q[3];
ry(-2.6814626475196786) q[7];
cx q[3],q[7];
ry(-2.7882396339496025) q[3];
ry(-3.1114127293646527) q[7];
cx q[3],q[7];
ry(-2.4749240080876187) q[4];
ry(1.47744677292845) q[5];
cx q[4],q[5];
ry(1.1797487246349858) q[4];
ry(2.6719233274958483) q[5];
cx q[4],q[5];
ry(1.9598204200636964) q[4];
ry(-1.2004890045986079) q[6];
cx q[4],q[6];
ry(2.359704264203106) q[4];
ry(2.015073601711939) q[6];
cx q[4],q[6];
ry(3.1150816861690616) q[4];
ry(-2.6853269017831582) q[7];
cx q[4],q[7];
ry(-3.086707727331386) q[4];
ry(1.2668023828465929) q[7];
cx q[4],q[7];
ry(1.2089380105103738) q[5];
ry(-3.070146758151357) q[6];
cx q[5],q[6];
ry(-0.44972047795141223) q[5];
ry(-0.7158023189575591) q[6];
cx q[5],q[6];
ry(-0.5236230301821436) q[5];
ry(-2.177507992312334) q[7];
cx q[5],q[7];
ry(-0.4296949498293171) q[5];
ry(-1.2707286186162996) q[7];
cx q[5],q[7];
ry(3.127093202881278) q[6];
ry(0.9292306974772524) q[7];
cx q[6],q[7];
ry(2.661257934913352) q[6];
ry(-0.9488691692072523) q[7];
cx q[6],q[7];
ry(-1.5587224270887372) q[0];
ry(-0.6729740314155697) q[1];
cx q[0],q[1];
ry(2.381209621740865) q[0];
ry(-0.011435357799350017) q[1];
cx q[0],q[1];
ry(2.7894051822553085) q[0];
ry(0.72593240720354) q[2];
cx q[0],q[2];
ry(1.1726150852843364) q[0];
ry(2.5397752947626566) q[2];
cx q[0],q[2];
ry(3.066060275569616) q[0];
ry(0.22017030862237075) q[3];
cx q[0],q[3];
ry(-2.528256706723836) q[0];
ry(-0.08896373078275044) q[3];
cx q[0],q[3];
ry(0.6159665357850761) q[0];
ry(-1.67996986102601) q[4];
cx q[0],q[4];
ry(-2.242885684798338) q[0];
ry(-2.7759200547263254) q[4];
cx q[0],q[4];
ry(0.07566084122254324) q[0];
ry(-1.1792521604262802) q[5];
cx q[0],q[5];
ry(2.406933833113517) q[0];
ry(-1.8082048193163913) q[5];
cx q[0],q[5];
ry(2.070953666085975) q[0];
ry(-2.7292045436179064) q[6];
cx q[0],q[6];
ry(1.8429041445820948) q[0];
ry(-0.6535696645788998) q[6];
cx q[0],q[6];
ry(-2.179136337659839) q[0];
ry(1.4443276207937548) q[7];
cx q[0],q[7];
ry(-1.5517879830676966) q[0];
ry(-2.075777997808202) q[7];
cx q[0],q[7];
ry(1.3833414816590697) q[1];
ry(-0.0402205349537228) q[2];
cx q[1],q[2];
ry(0.5880975807848294) q[1];
ry(0.5366231789528095) q[2];
cx q[1],q[2];
ry(-0.9366888363042829) q[1];
ry(1.4754097202309617) q[3];
cx q[1],q[3];
ry(1.7988787809409115) q[1];
ry(1.0262720277641382) q[3];
cx q[1],q[3];
ry(2.2419440184932706) q[1];
ry(0.4630919833186917) q[4];
cx q[1],q[4];
ry(-1.2088021672036575) q[1];
ry(-1.160581457405963) q[4];
cx q[1],q[4];
ry(1.2994523559291578) q[1];
ry(2.2785699326728515) q[5];
cx q[1],q[5];
ry(1.6869290317064458) q[1];
ry(-0.9917340546444882) q[5];
cx q[1],q[5];
ry(1.7869334266140573) q[1];
ry(1.896521460362628) q[6];
cx q[1],q[6];
ry(-2.6435425279045357) q[1];
ry(-1.7269002580029094) q[6];
cx q[1],q[6];
ry(-2.251655567318924) q[1];
ry(2.1269429099020805) q[7];
cx q[1],q[7];
ry(0.5036926177365224) q[1];
ry(1.1530832099403394) q[7];
cx q[1],q[7];
ry(-1.5947976301486018) q[2];
ry(2.4358180372260856) q[3];
cx q[2],q[3];
ry(0.6050501051600214) q[2];
ry(0.5441309576295206) q[3];
cx q[2],q[3];
ry(1.7507156635044698) q[2];
ry(-2.368951914360407) q[4];
cx q[2],q[4];
ry(-2.0146531945179467) q[2];
ry(0.8353364546125305) q[4];
cx q[2],q[4];
ry(-1.3348047455449974) q[2];
ry(2.891275662669178) q[5];
cx q[2],q[5];
ry(-2.989868305231215) q[2];
ry(1.4601174264610277) q[5];
cx q[2],q[5];
ry(-0.17302210263072126) q[2];
ry(-1.9414336285472955) q[6];
cx q[2],q[6];
ry(-1.377218596994389) q[2];
ry(-2.1179821712589533) q[6];
cx q[2],q[6];
ry(-2.8123091473191355) q[2];
ry(-1.5067654503749868) q[7];
cx q[2],q[7];
ry(0.9284975588585569) q[2];
ry(3.115878240482926) q[7];
cx q[2],q[7];
ry(0.022481893883355374) q[3];
ry(2.065957224465865) q[4];
cx q[3],q[4];
ry(-0.6921743394777078) q[3];
ry(-0.6170160190329197) q[4];
cx q[3],q[4];
ry(3.021192681994864) q[3];
ry(-1.1141294073872574) q[5];
cx q[3],q[5];
ry(-0.5195860915185974) q[3];
ry(0.9954922060724057) q[5];
cx q[3],q[5];
ry(-1.2815118353637576) q[3];
ry(3.003042242909686) q[6];
cx q[3],q[6];
ry(-1.4385176996948752) q[3];
ry(-0.6991863520583559) q[6];
cx q[3],q[6];
ry(-0.524227120815616) q[3];
ry(2.087639791297014) q[7];
cx q[3],q[7];
ry(0.8740897636542668) q[3];
ry(0.9511226920258724) q[7];
cx q[3],q[7];
ry(-1.2693873388032384) q[4];
ry(-2.843224926204491) q[5];
cx q[4],q[5];
ry(-2.291626026689002) q[4];
ry(2.4998772574198465) q[5];
cx q[4],q[5];
ry(-3.006641045448464) q[4];
ry(2.864856077765429) q[6];
cx q[4],q[6];
ry(0.3886660568983489) q[4];
ry(-0.8108474126842308) q[6];
cx q[4],q[6];
ry(-0.270797037491644) q[4];
ry(3.1326130242215187) q[7];
cx q[4],q[7];
ry(-0.6439466574454249) q[4];
ry(0.7271608080421768) q[7];
cx q[4],q[7];
ry(-0.43806666823392465) q[5];
ry(2.611011421362089) q[6];
cx q[5],q[6];
ry(1.6457968523144588) q[5];
ry(2.6569484550446854) q[6];
cx q[5],q[6];
ry(2.460379550230929) q[5];
ry(2.971006880747584) q[7];
cx q[5],q[7];
ry(1.1943504242960374) q[5];
ry(0.2505208668081762) q[7];
cx q[5],q[7];
ry(-0.6852655073526795) q[6];
ry(1.9292207214758366) q[7];
cx q[6],q[7];
ry(1.8221704951655904) q[6];
ry(1.639061268317591) q[7];
cx q[6],q[7];
ry(1.130390198286368) q[0];
ry(1.2957140386030392) q[1];
cx q[0],q[1];
ry(0.13732229972437615) q[0];
ry(-3.059437430258065) q[1];
cx q[0],q[1];
ry(-1.0665507750737238) q[0];
ry(0.4260673501974477) q[2];
cx q[0],q[2];
ry(1.7421708567641234) q[0];
ry(2.852344636985512) q[2];
cx q[0],q[2];
ry(2.253340480353843) q[0];
ry(0.1421771493456223) q[3];
cx q[0],q[3];
ry(-2.688707527725339) q[0];
ry(0.049664705970930534) q[3];
cx q[0],q[3];
ry(2.3595933865956287) q[0];
ry(1.3429901770525894) q[4];
cx q[0],q[4];
ry(1.5688183145381025) q[0];
ry(2.703104790784135) q[4];
cx q[0],q[4];
ry(1.5520406236463566) q[0];
ry(2.40144771865013) q[5];
cx q[0],q[5];
ry(-1.0431201851804888) q[0];
ry(2.0748192744519187) q[5];
cx q[0],q[5];
ry(2.0931198346134208) q[0];
ry(-1.5581859171087276) q[6];
cx q[0],q[6];
ry(0.9053962551266479) q[0];
ry(-2.871581804950407) q[6];
cx q[0],q[6];
ry(-1.310634607945634) q[0];
ry(-1.2772608878573966) q[7];
cx q[0],q[7];
ry(2.4741032940404772) q[0];
ry(0.9650170572664419) q[7];
cx q[0],q[7];
ry(-2.3914992056492737) q[1];
ry(0.6568736646127151) q[2];
cx q[1],q[2];
ry(1.4656869975813045) q[1];
ry(-2.810585160057291) q[2];
cx q[1],q[2];
ry(-1.0586011128722213) q[1];
ry(-3.007381683923102) q[3];
cx q[1],q[3];
ry(0.7771180016106998) q[1];
ry(2.5069757108182373) q[3];
cx q[1],q[3];
ry(-2.3138476782195028) q[1];
ry(-0.839191756965544) q[4];
cx q[1],q[4];
ry(-0.12500579505596043) q[1];
ry(-0.5118763573196068) q[4];
cx q[1],q[4];
ry(-0.456650750708801) q[1];
ry(0.12050964883286142) q[5];
cx q[1],q[5];
ry(2.4677160067260733) q[1];
ry(1.0300648293988317) q[5];
cx q[1],q[5];
ry(3.0494952283257084) q[1];
ry(-3.0240344923343216) q[6];
cx q[1],q[6];
ry(-1.8851320937356497) q[1];
ry(0.9020769427383097) q[6];
cx q[1],q[6];
ry(2.1913962540873944) q[1];
ry(-1.3090702191142356) q[7];
cx q[1],q[7];
ry(-0.4834547405668454) q[1];
ry(2.406298137040637) q[7];
cx q[1],q[7];
ry(-1.8079637313290755) q[2];
ry(1.6051933222501296) q[3];
cx q[2],q[3];
ry(0.8426450516596935) q[2];
ry(-3.0747836648665046) q[3];
cx q[2],q[3];
ry(0.38024944248597414) q[2];
ry(2.4676436931630623) q[4];
cx q[2],q[4];
ry(2.931936179636063) q[2];
ry(0.09260966115961491) q[4];
cx q[2],q[4];
ry(2.592592026455127) q[2];
ry(-2.6328463628537713) q[5];
cx q[2],q[5];
ry(-0.5838548827087512) q[2];
ry(-0.20378622522725195) q[5];
cx q[2],q[5];
ry(-2.2565905650108085) q[2];
ry(1.7776957574250811) q[6];
cx q[2],q[6];
ry(3.0745152594736225) q[2];
ry(-1.7678037460450835) q[6];
cx q[2],q[6];
ry(2.69680907774402) q[2];
ry(2.049246352420817) q[7];
cx q[2],q[7];
ry(1.0075659919349933) q[2];
ry(-2.201064363259425) q[7];
cx q[2],q[7];
ry(-0.7043578857881929) q[3];
ry(0.5264024361778787) q[4];
cx q[3],q[4];
ry(-1.3315618735199186) q[3];
ry(-0.5632384561653742) q[4];
cx q[3],q[4];
ry(2.6123832773905944) q[3];
ry(-2.9348922299279754) q[5];
cx q[3],q[5];
ry(2.8848567732448127) q[3];
ry(1.7796294825566379) q[5];
cx q[3],q[5];
ry(-0.6414600617353707) q[3];
ry(1.4802984886325044) q[6];
cx q[3],q[6];
ry(-1.2707569909454617) q[3];
ry(-1.3661308318047132) q[6];
cx q[3],q[6];
ry(0.5929265829732211) q[3];
ry(0.30224003930885024) q[7];
cx q[3],q[7];
ry(1.672002451350373) q[3];
ry(1.4869402423868447) q[7];
cx q[3],q[7];
ry(1.897613837435781) q[4];
ry(-0.20199399785930378) q[5];
cx q[4],q[5];
ry(2.8747477879571846) q[4];
ry(2.4818866149312035) q[5];
cx q[4],q[5];
ry(1.9655872358760684) q[4];
ry(-2.952674683715417) q[6];
cx q[4],q[6];
ry(-2.5298157842403453) q[4];
ry(0.23441544193546449) q[6];
cx q[4],q[6];
ry(-3.137199926103319) q[4];
ry(-2.6503228193953348) q[7];
cx q[4],q[7];
ry(0.2020360909094574) q[4];
ry(-2.6777434040810313) q[7];
cx q[4],q[7];
ry(1.6713970240341143) q[5];
ry(-0.2648308962411354) q[6];
cx q[5],q[6];
ry(2.40381645664317) q[5];
ry(-0.41504600842774586) q[6];
cx q[5],q[6];
ry(2.6384065737272833) q[5];
ry(-1.707362568597176) q[7];
cx q[5],q[7];
ry(0.9865073255170685) q[5];
ry(0.3193066218127561) q[7];
cx q[5],q[7];
ry(-1.508314070007706) q[6];
ry(0.3478431107702882) q[7];
cx q[6],q[7];
ry(-1.1651634039347873) q[6];
ry(1.5122354963515752) q[7];
cx q[6],q[7];
ry(2.2923724973931576) q[0];
ry(2.4445178030813213) q[1];
cx q[0],q[1];
ry(-1.5543795212164422) q[0];
ry(0.33096608510762476) q[1];
cx q[0],q[1];
ry(-0.57796992589702) q[0];
ry(2.4486312323009827) q[2];
cx q[0],q[2];
ry(1.642848571300797) q[0];
ry(0.7539156258853179) q[2];
cx q[0],q[2];
ry(2.4734933922255684) q[0];
ry(-2.4269550896157885) q[3];
cx q[0],q[3];
ry(0.7905271048257596) q[0];
ry(0.23131019356624238) q[3];
cx q[0],q[3];
ry(1.0043307594766668) q[0];
ry(-0.08791447214334396) q[4];
cx q[0],q[4];
ry(-0.213911322040675) q[0];
ry(-2.4376236171232493) q[4];
cx q[0],q[4];
ry(-2.614789989748998) q[0];
ry(-2.029013417684898) q[5];
cx q[0],q[5];
ry(0.4481387524443914) q[0];
ry(0.9421348178381412) q[5];
cx q[0],q[5];
ry(-2.4669607241203626) q[0];
ry(-2.54856731125593) q[6];
cx q[0],q[6];
ry(1.9546009165294267) q[0];
ry(2.8070484032783596) q[6];
cx q[0],q[6];
ry(-2.9582419944295975) q[0];
ry(1.455410822253188) q[7];
cx q[0],q[7];
ry(-1.2649823581479183) q[0];
ry(1.406756054908976) q[7];
cx q[0],q[7];
ry(2.6732483774186786) q[1];
ry(-2.642155921693602) q[2];
cx q[1],q[2];
ry(-2.30270579059593) q[1];
ry(-1.3404267765504514) q[2];
cx q[1],q[2];
ry(-2.062445492918944) q[1];
ry(-3.027983524351872) q[3];
cx q[1],q[3];
ry(-0.4789661743724638) q[1];
ry(-0.4014973022695232) q[3];
cx q[1],q[3];
ry(-0.6613371690972755) q[1];
ry(2.1128604769249124) q[4];
cx q[1],q[4];
ry(-0.750804475271435) q[1];
ry(-2.256751172950394) q[4];
cx q[1],q[4];
ry(0.13329366693611444) q[1];
ry(-0.8917310041856137) q[5];
cx q[1],q[5];
ry(2.1168686077074232) q[1];
ry(-2.756097665459139) q[5];
cx q[1],q[5];
ry(2.668141113393688) q[1];
ry(0.4270266850532247) q[6];
cx q[1],q[6];
ry(-1.245884637603785) q[1];
ry(-1.0704023028053138) q[6];
cx q[1],q[6];
ry(-1.719077225619328) q[1];
ry(3.0512511567723912) q[7];
cx q[1],q[7];
ry(1.4494320876383568) q[1];
ry(-0.08179486931447306) q[7];
cx q[1],q[7];
ry(0.18409026631103417) q[2];
ry(-1.8284454486833788) q[3];
cx q[2],q[3];
ry(-3.1355647277001757) q[2];
ry(1.58784991899699) q[3];
cx q[2],q[3];
ry(0.5483197970901291) q[2];
ry(-2.3910879259734847) q[4];
cx q[2],q[4];
ry(-2.8925817902661284) q[2];
ry(0.8009560829394964) q[4];
cx q[2],q[4];
ry(-2.1070813778190836) q[2];
ry(-1.2326573395933096) q[5];
cx q[2],q[5];
ry(-1.6752467222341783) q[2];
ry(2.324972135188902) q[5];
cx q[2],q[5];
ry(-2.9269656815840057) q[2];
ry(-0.12419257259441221) q[6];
cx q[2],q[6];
ry(-2.320941671485845) q[2];
ry(1.0519254256573758) q[6];
cx q[2],q[6];
ry(1.0649647680475436) q[2];
ry(1.8768563863329217) q[7];
cx q[2],q[7];
ry(-0.5077717953483463) q[2];
ry(-1.2324727600662897) q[7];
cx q[2],q[7];
ry(-0.8586366308325388) q[3];
ry(-1.9296988990654262) q[4];
cx q[3],q[4];
ry(0.8345170168595882) q[3];
ry(0.22832190027396582) q[4];
cx q[3],q[4];
ry(-2.635618156195465) q[3];
ry(0.40289503140352156) q[5];
cx q[3],q[5];
ry(1.406105265729342) q[3];
ry(0.36134936258439937) q[5];
cx q[3],q[5];
ry(-2.272876234127935) q[3];
ry(-3.0986329233460776) q[6];
cx q[3],q[6];
ry(2.6692106897034718) q[3];
ry(1.9089881494302468) q[6];
cx q[3],q[6];
ry(3.1206782860950977) q[3];
ry(-2.505812031902557) q[7];
cx q[3],q[7];
ry(-1.975959386081272) q[3];
ry(0.34887263642181043) q[7];
cx q[3],q[7];
ry(0.3323926512886457) q[4];
ry(0.36938263564231555) q[5];
cx q[4],q[5];
ry(2.79018566195043) q[4];
ry(2.2640133137323692) q[5];
cx q[4],q[5];
ry(-2.336379618051846) q[4];
ry(2.670480849373615) q[6];
cx q[4],q[6];
ry(-1.4858547039733807) q[4];
ry(0.4502055091567811) q[6];
cx q[4],q[6];
ry(1.4717755455010924) q[4];
ry(2.4696612764064354) q[7];
cx q[4],q[7];
ry(-2.9099024496201102) q[4];
ry(-2.522224103962632) q[7];
cx q[4],q[7];
ry(1.6913327079831963) q[5];
ry(2.55913154085732) q[6];
cx q[5],q[6];
ry(-0.9964710868524607) q[5];
ry(-0.627353758297751) q[6];
cx q[5],q[6];
ry(-1.6104007569436378) q[5];
ry(1.733360193899324) q[7];
cx q[5],q[7];
ry(-0.2537213425221237) q[5];
ry(-1.4224766669257063) q[7];
cx q[5],q[7];
ry(1.6747828019231843) q[6];
ry(-2.4009515357032902) q[7];
cx q[6],q[7];
ry(0.9106459167766117) q[6];
ry(0.8510469394849357) q[7];
cx q[6],q[7];
ry(-0.4673164700141816) q[0];
ry(1.291647494116309) q[1];
cx q[0],q[1];
ry(-2.3756811408522926) q[0];
ry(2.606348184858647) q[1];
cx q[0],q[1];
ry(-2.871695084286909) q[0];
ry(1.6821205255337803) q[2];
cx q[0],q[2];
ry(-2.326612982820813) q[0];
ry(-3.000800772042218) q[2];
cx q[0],q[2];
ry(1.1574792405699046) q[0];
ry(2.8945508030252163) q[3];
cx q[0],q[3];
ry(-0.44528325327936746) q[0];
ry(1.120136122467036) q[3];
cx q[0],q[3];
ry(-1.336973228599189) q[0];
ry(-1.7459047612719896) q[4];
cx q[0],q[4];
ry(-2.0471070843182657) q[0];
ry(-3.008330245070965) q[4];
cx q[0],q[4];
ry(0.5967683246481004) q[0];
ry(-1.8237022910966507) q[5];
cx q[0],q[5];
ry(0.7474212459533436) q[0];
ry(-0.603671208044021) q[5];
cx q[0],q[5];
ry(-2.741439838697288) q[0];
ry(-0.33491859016989345) q[6];
cx q[0],q[6];
ry(-2.631127139646728) q[0];
ry(2.5362519736055713) q[6];
cx q[0],q[6];
ry(1.4759523860661177) q[0];
ry(-2.3599601284448157) q[7];
cx q[0],q[7];
ry(2.055867898596968) q[0];
ry(-0.5030224585251809) q[7];
cx q[0],q[7];
ry(-0.587271057820897) q[1];
ry(-2.92759818411569) q[2];
cx q[1],q[2];
ry(3.0579330621490537) q[1];
ry(-2.933915297069388) q[2];
cx q[1],q[2];
ry(1.060805984854765) q[1];
ry(1.0642444087628466) q[3];
cx q[1],q[3];
ry(-2.8773027307604138) q[1];
ry(2.191516497408677) q[3];
cx q[1],q[3];
ry(-1.960121355795092) q[1];
ry(1.0509153669337632) q[4];
cx q[1],q[4];
ry(-2.8776722325774657) q[1];
ry(0.515850752593704) q[4];
cx q[1],q[4];
ry(-1.4034939064578884) q[1];
ry(-0.0005231724977269932) q[5];
cx q[1],q[5];
ry(2.2639214623169743) q[1];
ry(0.5751337782559548) q[5];
cx q[1],q[5];
ry(-2.7189942688883058) q[1];
ry(-0.44224928083736764) q[6];
cx q[1],q[6];
ry(-1.712802714371922) q[1];
ry(-0.8224815669595743) q[6];
cx q[1],q[6];
ry(0.18520682291859503) q[1];
ry(-0.4507639389691697) q[7];
cx q[1],q[7];
ry(1.0731243551816882) q[1];
ry(0.3375164938433228) q[7];
cx q[1],q[7];
ry(0.8300838412746573) q[2];
ry(0.5287834792261109) q[3];
cx q[2],q[3];
ry(-1.4904055274831882) q[2];
ry(1.5139131450013321) q[3];
cx q[2],q[3];
ry(-0.09452215729230033) q[2];
ry(-1.3828895389703846) q[4];
cx q[2],q[4];
ry(-2.022819942822778) q[2];
ry(2.4054548004367793) q[4];
cx q[2],q[4];
ry(0.9044109314502625) q[2];
ry(0.012079524196814795) q[5];
cx q[2],q[5];
ry(2.4207696089693425) q[2];
ry(-1.91062136648216) q[5];
cx q[2],q[5];
ry(2.1821494097461773) q[2];
ry(0.6496417715942617) q[6];
cx q[2],q[6];
ry(0.15008421647215542) q[2];
ry(-1.052537539996389) q[6];
cx q[2],q[6];
ry(1.5412188846789174) q[2];
ry(-0.8014592113160051) q[7];
cx q[2],q[7];
ry(-2.6728025833541684) q[2];
ry(0.09232287433508053) q[7];
cx q[2],q[7];
ry(-1.1020239522869266) q[3];
ry(1.6705615596171164) q[4];
cx q[3],q[4];
ry(-2.3232842958487403) q[3];
ry(-1.663484104926501) q[4];
cx q[3],q[4];
ry(-0.9184145485139884) q[3];
ry(0.8012946947575794) q[5];
cx q[3],q[5];
ry(-2.7330564776020454) q[3];
ry(0.9674920565056038) q[5];
cx q[3],q[5];
ry(-2.8913209181540824) q[3];
ry(-2.064097406662195) q[6];
cx q[3],q[6];
ry(-0.5887687116605891) q[3];
ry(2.7143588857882266) q[6];
cx q[3],q[6];
ry(-3.023468885106986) q[3];
ry(-1.0264213971613536) q[7];
cx q[3],q[7];
ry(-0.24149806179158512) q[3];
ry(-0.974827103337585) q[7];
cx q[3],q[7];
ry(-2.580212164476158) q[4];
ry(0.5979821544631676) q[5];
cx q[4],q[5];
ry(-0.28663122354736004) q[4];
ry(-1.884528128098177) q[5];
cx q[4],q[5];
ry(-0.43678787661775065) q[4];
ry(-1.6930865371453176) q[6];
cx q[4],q[6];
ry(-1.9067990891007631) q[4];
ry(1.8200487840743298) q[6];
cx q[4],q[6];
ry(2.81906277015357) q[4];
ry(2.8295364777296803) q[7];
cx q[4],q[7];
ry(-2.178829542526254) q[4];
ry(-2.3917156470566896) q[7];
cx q[4],q[7];
ry(-2.29042017460466) q[5];
ry(-1.429513248938707) q[6];
cx q[5],q[6];
ry(-2.860611708409812) q[5];
ry(-0.15945344300620987) q[6];
cx q[5],q[6];
ry(1.203483702820718) q[5];
ry(-2.048144650396464) q[7];
cx q[5],q[7];
ry(-0.4745296791464147) q[5];
ry(2.3619132998912225) q[7];
cx q[5],q[7];
ry(1.8088269255369511) q[6];
ry(-1.510925609703183) q[7];
cx q[6],q[7];
ry(2.533105301211569) q[6];
ry(-0.21356386197689758) q[7];
cx q[6],q[7];
ry(-2.873277624513759) q[0];
ry(1.4740266409347207) q[1];
ry(2.403823152621944) q[2];
ry(2.626938968119601) q[3];
ry(1.9960434344839664) q[4];
ry(-0.3072261217478266) q[5];
ry(-0.22883169380878599) q[6];
ry(-2.9432176704804913) q[7];