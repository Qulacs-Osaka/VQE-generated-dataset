OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.35644010384999536) q[0];
ry(0.4674159223649282) q[1];
cx q[0],q[1];
ry(-3.1059435149827257) q[0];
ry(-1.824836452535264) q[1];
cx q[0],q[1];
ry(1.3559423644178676) q[1];
ry(-2.9107030830077703) q[2];
cx q[1],q[2];
ry(1.2804467442968808) q[1];
ry(0.6442809778736143) q[2];
cx q[1],q[2];
ry(0.2653460279194314) q[2];
ry(-0.8024523151304261) q[3];
cx q[2],q[3];
ry(0.3911258180511989) q[2];
ry(-1.798608685222999) q[3];
cx q[2],q[3];
ry(-1.0014439986953754) q[0];
ry(-2.1715590747090348) q[1];
cx q[0],q[1];
ry(-1.4171070174085711) q[0];
ry(-2.740632956725259) q[1];
cx q[0],q[1];
ry(-1.4012429627856946) q[1];
ry(-0.5410776229731118) q[2];
cx q[1],q[2];
ry(-2.9386112382288565) q[1];
ry(-1.693016297979476) q[2];
cx q[1],q[2];
ry(-0.6732625067103841) q[2];
ry(1.177909755451425) q[3];
cx q[2],q[3];
ry(-2.905415088575163) q[2];
ry(3.079276276485773) q[3];
cx q[2],q[3];
ry(1.848316826505312) q[0];
ry(-2.395277001660462) q[1];
cx q[0],q[1];
ry(-2.117803030044126) q[0];
ry(1.4667184733209533) q[1];
cx q[0],q[1];
ry(-2.5096401789317193) q[1];
ry(-2.7889440836702395) q[2];
cx q[1],q[2];
ry(-3.0725445839115757) q[1];
ry(2.2914502235955028) q[2];
cx q[1],q[2];
ry(0.79950427147738) q[2];
ry(0.8902331208878634) q[3];
cx q[2],q[3];
ry(-0.6581855866963808) q[2];
ry(2.9265988494880655) q[3];
cx q[2],q[3];
ry(-3.0437651033269986) q[0];
ry(-0.7780188079086161) q[1];
cx q[0],q[1];
ry(1.748577225264471) q[0];
ry(-1.0714709310156245) q[1];
cx q[0],q[1];
ry(2.606817323528502) q[1];
ry(-1.4359081896260848) q[2];
cx q[1],q[2];
ry(1.258000835568016) q[1];
ry(-0.823602022943305) q[2];
cx q[1],q[2];
ry(1.1782434147777594) q[2];
ry(1.2818365624076566) q[3];
cx q[2],q[3];
ry(1.9639131620101136) q[2];
ry(2.257401952848234) q[3];
cx q[2],q[3];
ry(1.3138700560156966) q[0];
ry(-2.1104207902454974) q[1];
cx q[0],q[1];
ry(-2.3020360101371073) q[0];
ry(-0.6364917280573996) q[1];
cx q[0],q[1];
ry(-1.7092152744829372) q[1];
ry(-2.5427961749416825) q[2];
cx q[1],q[2];
ry(1.0292445793441418) q[1];
ry(0.2763152907276858) q[2];
cx q[1],q[2];
ry(-1.9563481294276848) q[2];
ry(1.059841601187829) q[3];
cx q[2],q[3];
ry(-1.6908183260390162) q[2];
ry(2.5694284998095704) q[3];
cx q[2],q[3];
ry(2.403771052133826) q[0];
ry(1.1131623855879937) q[1];
cx q[0],q[1];
ry(-2.0071772149333653) q[0];
ry(-1.1318551961291101) q[1];
cx q[0],q[1];
ry(0.8839151437029809) q[1];
ry(2.091046700869991) q[2];
cx q[1],q[2];
ry(-0.1382665725934999) q[1];
ry(-1.8860133194036337) q[2];
cx q[1],q[2];
ry(-2.9276623026217203) q[2];
ry(-0.07476386936957825) q[3];
cx q[2],q[3];
ry(-0.5446720836773591) q[2];
ry(-0.7983218183403402) q[3];
cx q[2],q[3];
ry(3.0480016470814255) q[0];
ry(-0.0025088014461362904) q[1];
cx q[0],q[1];
ry(1.4082558226999242) q[0];
ry(-0.3888156972014927) q[1];
cx q[0],q[1];
ry(-0.3344283890000464) q[1];
ry(1.6918791335365153) q[2];
cx q[1],q[2];
ry(1.0602183257169635) q[1];
ry(2.1302353222177963) q[2];
cx q[1],q[2];
ry(1.221650184601841) q[2];
ry(-1.3652179442019683) q[3];
cx q[2],q[3];
ry(-2.855753420604637) q[2];
ry(-0.10446972620536776) q[3];
cx q[2],q[3];
ry(1.6800551928664145) q[0];
ry(-0.9860044357079087) q[1];
cx q[0],q[1];
ry(1.2355267553853235) q[0];
ry(2.122554655307651) q[1];
cx q[0],q[1];
ry(-1.9858184679356483) q[1];
ry(-2.0002853449208247) q[2];
cx q[1],q[2];
ry(-0.24289770271186656) q[1];
ry(-0.39163554913557785) q[2];
cx q[1],q[2];
ry(-2.1133603986890446) q[2];
ry(-1.2947158556260758) q[3];
cx q[2],q[3];
ry(-2.0476795225370443) q[2];
ry(2.9037939191293254) q[3];
cx q[2],q[3];
ry(-0.8770082254594567) q[0];
ry(-0.46562012166133604) q[1];
cx q[0],q[1];
ry(2.5565238188142727) q[0];
ry(0.8951653000879176) q[1];
cx q[0],q[1];
ry(0.47404546702820394) q[1];
ry(-0.48112137729492715) q[2];
cx q[1],q[2];
ry(0.8399549530450029) q[1];
ry(0.9075408348800487) q[2];
cx q[1],q[2];
ry(2.963074489695984) q[2];
ry(1.6231988016524097) q[3];
cx q[2],q[3];
ry(2.437808320529347) q[2];
ry(-1.314267348572586) q[3];
cx q[2],q[3];
ry(3.1214123763280366) q[0];
ry(-1.9035381739666501) q[1];
cx q[0],q[1];
ry(1.1716195910206961) q[0];
ry(-1.2171362412990685) q[1];
cx q[0],q[1];
ry(-2.631422414898561) q[1];
ry(-1.0522702908392507) q[2];
cx q[1],q[2];
ry(-2.159522471653466) q[1];
ry(2.117780713099358) q[2];
cx q[1],q[2];
ry(0.4570720419582504) q[2];
ry(-2.8951608725400706) q[3];
cx q[2],q[3];
ry(1.2406365204270746) q[2];
ry(2.515101458953786) q[3];
cx q[2],q[3];
ry(0.6790132375659725) q[0];
ry(0.18573583583267153) q[1];
cx q[0],q[1];
ry(-2.3113202919947837) q[0];
ry(0.13409840777384563) q[1];
cx q[0],q[1];
ry(-0.09452815444950423) q[1];
ry(0.6830340464855954) q[2];
cx q[1],q[2];
ry(-0.2265986377370071) q[1];
ry(-0.9991766361693042) q[2];
cx q[1],q[2];
ry(1.3898227011158433) q[2];
ry(-2.589765212171872) q[3];
cx q[2],q[3];
ry(-1.1869890297999799) q[2];
ry(-0.707562130875279) q[3];
cx q[2],q[3];
ry(1.073762128613512) q[0];
ry(-0.5637441603783362) q[1];
cx q[0],q[1];
ry(-2.9242965263335146) q[0];
ry(-3.0714739121086336) q[1];
cx q[0],q[1];
ry(-1.6546756010666936) q[1];
ry(0.02188846116135449) q[2];
cx q[1],q[2];
ry(-3.12445784363578) q[1];
ry(0.7233278859109181) q[2];
cx q[1],q[2];
ry(2.4213107664127547) q[2];
ry(-2.1960699626328504) q[3];
cx q[2],q[3];
ry(-1.74239441838382) q[2];
ry(1.8141175210560156) q[3];
cx q[2],q[3];
ry(-3.0174558634421746) q[0];
ry(1.711090665659335) q[1];
cx q[0],q[1];
ry(-1.8079592787608547) q[0];
ry(-0.02796706022527129) q[1];
cx q[0],q[1];
ry(-0.6941489285498309) q[1];
ry(1.7179386350550945) q[2];
cx q[1],q[2];
ry(-0.6161356564232287) q[1];
ry(0.17776456372777355) q[2];
cx q[1],q[2];
ry(0.1430874512125799) q[2];
ry(-0.3431840569886731) q[3];
cx q[2],q[3];
ry(0.16569983023212473) q[2];
ry(2.6387443027954927) q[3];
cx q[2],q[3];
ry(2.9989823467716854) q[0];
ry(2.1744741300670025) q[1];
cx q[0],q[1];
ry(1.4487814900846614) q[0];
ry(0.008059620614184304) q[1];
cx q[0],q[1];
ry(-0.3816991933522109) q[1];
ry(2.6644367310945745) q[2];
cx q[1],q[2];
ry(-1.9723709917818253) q[1];
ry(0.5148354180163245) q[2];
cx q[1],q[2];
ry(-2.1361031405078754) q[2];
ry(-2.4050820933942947) q[3];
cx q[2],q[3];
ry(-2.6635277976116196) q[2];
ry(1.137706301652779) q[3];
cx q[2],q[3];
ry(-0.18188950336811197) q[0];
ry(-0.48158257061834575) q[1];
cx q[0],q[1];
ry(-0.4270878855453297) q[0];
ry(-2.5384746891721366) q[1];
cx q[0],q[1];
ry(1.3112161567690581) q[1];
ry(-0.19286498268675098) q[2];
cx q[1],q[2];
ry(-2.804108516648869) q[1];
ry(0.4192618957855537) q[2];
cx q[1],q[2];
ry(2.9764327791138974) q[2];
ry(1.625681878646139) q[3];
cx q[2],q[3];
ry(1.5596043005707028) q[2];
ry(-1.54584255583009) q[3];
cx q[2],q[3];
ry(2.277412825142382) q[0];
ry(-1.1571890182104925) q[1];
cx q[0],q[1];
ry(-0.044795624213616166) q[0];
ry(0.10818528142955099) q[1];
cx q[0],q[1];
ry(0.6572822038303494) q[1];
ry(-0.2772190271621282) q[2];
cx q[1],q[2];
ry(2.0149821216444055) q[1];
ry(1.377161756481191) q[2];
cx q[1],q[2];
ry(2.7241489768169167) q[2];
ry(-1.7965686532827887) q[3];
cx q[2],q[3];
ry(1.21490917752885) q[2];
ry(1.9155750083516043) q[3];
cx q[2],q[3];
ry(-0.03466639909629147) q[0];
ry(-0.7707443503984993) q[1];
cx q[0],q[1];
ry(-0.512275647782988) q[0];
ry(-1.6400703884597485) q[1];
cx q[0],q[1];
ry(1.2482352587767733) q[1];
ry(-0.6366524491194063) q[2];
cx q[1],q[2];
ry(2.6159003355161965) q[1];
ry(1.7763342397914057) q[2];
cx q[1],q[2];
ry(-1.2951750467331413) q[2];
ry(-2.2980717832536204) q[3];
cx q[2],q[3];
ry(1.747113074010606) q[2];
ry(-2.538714949990193) q[3];
cx q[2],q[3];
ry(-2.1899613836794303) q[0];
ry(-1.7984473057615187) q[1];
cx q[0],q[1];
ry(-0.04431889986675053) q[0];
ry(-1.1081174103484979) q[1];
cx q[0],q[1];
ry(2.2199444196183897) q[1];
ry(3.040337457297178) q[2];
cx q[1],q[2];
ry(2.5115145134573753) q[1];
ry(0.36007819174851896) q[2];
cx q[1],q[2];
ry(-0.40540580572226936) q[2];
ry(-2.8971589385458922) q[3];
cx q[2],q[3];
ry(-0.22804116992389553) q[2];
ry(2.9232215262106) q[3];
cx q[2],q[3];
ry(-1.7532222837586542) q[0];
ry(1.1228872421098401) q[1];
cx q[0],q[1];
ry(-0.6535925805702174) q[0];
ry(1.2272631410219903) q[1];
cx q[0],q[1];
ry(1.6805012744062948) q[1];
ry(2.9624919581973224) q[2];
cx q[1],q[2];
ry(0.6271327560791544) q[1];
ry(-2.389054271274191) q[2];
cx q[1],q[2];
ry(0.9822827912952269) q[2];
ry(-0.06949139889539512) q[3];
cx q[2],q[3];
ry(0.6446884049200715) q[2];
ry(-0.1431148242272977) q[3];
cx q[2],q[3];
ry(1.6318123302801206) q[0];
ry(1.9470800945805662) q[1];
cx q[0],q[1];
ry(1.7156153607381448) q[0];
ry(-2.910873700401823) q[1];
cx q[0],q[1];
ry(-0.2450507081467335) q[1];
ry(-0.0018601914694178492) q[2];
cx q[1],q[2];
ry(-1.377559650349696) q[1];
ry(-0.07942014929476923) q[2];
cx q[1],q[2];
ry(-2.6512635388444843) q[2];
ry(-0.29592569900121646) q[3];
cx q[2],q[3];
ry(2.0247890911499806) q[2];
ry(-3.063918059870694) q[3];
cx q[2],q[3];
ry(-2.4987817743141467) q[0];
ry(0.6831959657032125) q[1];
cx q[0],q[1];
ry(-2.2068736396589004) q[0];
ry(2.629510735121438) q[1];
cx q[0],q[1];
ry(-1.959034707991833) q[1];
ry(-2.313682227540254) q[2];
cx q[1],q[2];
ry(-1.7790336573513823) q[1];
ry(1.3522360685874053) q[2];
cx q[1],q[2];
ry(-3.1306485101290966) q[2];
ry(-0.42985951381490584) q[3];
cx q[2],q[3];
ry(-1.312999056465494) q[2];
ry(-1.0476164566862334) q[3];
cx q[2],q[3];
ry(-2.4908663477097446) q[0];
ry(0.08595960701291894) q[1];
cx q[0],q[1];
ry(-3.112195972267056) q[0];
ry(2.7743969527193335) q[1];
cx q[0],q[1];
ry(-2.09790307733153) q[1];
ry(0.5888855042536143) q[2];
cx q[1],q[2];
ry(2.568476358566758) q[1];
ry(-0.8346102557367399) q[2];
cx q[1],q[2];
ry(3.094399388227766) q[2];
ry(0.29792904320052305) q[3];
cx q[2],q[3];
ry(3.0254594911310617) q[2];
ry(-0.9839002724210077) q[3];
cx q[2],q[3];
ry(-1.9161616576967866) q[0];
ry(2.8447834089342785) q[1];
cx q[0],q[1];
ry(1.6420211627669332) q[0];
ry(-1.959158489565881) q[1];
cx q[0],q[1];
ry(2.8605444093294916) q[1];
ry(0.1171764778873774) q[2];
cx q[1],q[2];
ry(0.3459281090980992) q[1];
ry(3.0919478420033513) q[2];
cx q[1],q[2];
ry(-2.1368070036148925) q[2];
ry(-1.7116441820608392) q[3];
cx q[2],q[3];
ry(-2.2743447371402405) q[2];
ry(2.3426590585918095) q[3];
cx q[2],q[3];
ry(1.9516706018475718) q[0];
ry(-0.16073055124832614) q[1];
cx q[0],q[1];
ry(-2.8480857696992423) q[0];
ry(-1.3839633401013618) q[1];
cx q[0],q[1];
ry(-2.3307817921698426) q[1];
ry(2.3737269983594502) q[2];
cx q[1],q[2];
ry(1.7937187887631705) q[1];
ry(-2.9079758857658935) q[2];
cx q[1],q[2];
ry(0.33443025675773264) q[2];
ry(0.809309820995514) q[3];
cx q[2],q[3];
ry(2.792768359278631) q[2];
ry(-2.5490082204992537) q[3];
cx q[2],q[3];
ry(2.2978335531277794) q[0];
ry(-0.23800716847859785) q[1];
ry(0.8843977111709416) q[2];
ry(-0.7426081208453086) q[3];