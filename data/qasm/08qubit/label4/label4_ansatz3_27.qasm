OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.405440537654248) q[0];
rz(0.22697115058324613) q[0];
ry(-2.5509344001281873) q[1];
rz(-1.5693218428462246) q[1];
ry(-1.75415979674115) q[2];
rz(1.4198804209799132) q[2];
ry(-1.559362754729551) q[3];
rz(-1.313060825555286) q[3];
ry(-2.8434921942638804) q[4];
rz(1.0039975134362802) q[4];
ry(-0.4636696178070898) q[5];
rz(0.12800988893346865) q[5];
ry(-0.9358668073309263) q[6];
rz(-0.7683673003761502) q[6];
ry(-1.5394464573317643) q[7];
rz(-1.1494482368179646) q[7];
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
ry(1.4323534405242029) q[0];
rz(-1.6489659279508304) q[0];
ry(-0.8268027758634418) q[1];
rz(2.5025122731967437) q[1];
ry(2.5106982257924506) q[2];
rz(-1.4920219475233132) q[2];
ry(-2.2581005476134886) q[3];
rz(0.06445795877992655) q[3];
ry(2.3341595357716276) q[4];
rz(-1.1603183885178057) q[4];
ry(-1.7024957061393475) q[5];
rz(-3.0423340722773897) q[5];
ry(-0.3496688370404947) q[6];
rz(3.0551193495388387) q[6];
ry(2.8441744563308573) q[7];
rz(2.8802905648334574) q[7];
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
ry(-2.4010208742963894) q[0];
rz(-2.1401121901852163) q[0];
ry(1.0451558980498965) q[1];
rz(2.7861144541530956) q[1];
ry(3.0821226213291992) q[2];
rz(-0.7084161900223577) q[2];
ry(-1.8214443553341164) q[3];
rz(1.5349570438423745) q[3];
ry(1.7944438763601542) q[4];
rz(2.6109753973845002) q[4];
ry(1.659056420015479) q[5];
rz(-0.8351758336789139) q[5];
ry(0.21037432536620138) q[6];
rz(1.6704703183786114) q[6];
ry(0.32789767150611215) q[7];
rz(1.0167468422420105) q[7];
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
ry(2.44313194166767) q[0];
rz(2.333047227081579) q[0];
ry(-3.063972211731207) q[1];
rz(-0.1997417660720462) q[1];
ry(-0.2379062082526264) q[2];
rz(-0.6976453562280435) q[2];
ry(0.17666466644373635) q[3];
rz(2.5618645779712264) q[3];
ry(-2.7596638276276892) q[4];
rz(0.6266869270836041) q[4];
ry(1.8235433918369013) q[5];
rz(-0.2886675384441965) q[5];
ry(1.0431318617747625) q[6];
rz(-2.1614576328548676) q[6];
ry(-2.8170567229923598) q[7];
rz(1.4816588802630986) q[7];
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
ry(2.6008739688880493) q[0];
rz(1.325928642831995) q[0];
ry(-0.3904520420603284) q[1];
rz(0.7465021996053909) q[1];
ry(2.002173152004602) q[2];
rz(2.4704495114014318) q[2];
ry(0.3328268339241012) q[3];
rz(-1.9817934823773484) q[3];
ry(2.0679475998633663) q[4];
rz(-2.455833767241398) q[4];
ry(-2.1466099858071885) q[5];
rz(-1.0578933036700562) q[5];
ry(3.0779810292580327) q[6];
rz(-1.5849265128391512) q[6];
ry(1.468696660377594) q[7];
rz(-1.1611818849490438) q[7];
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
ry(2.5182129893468206) q[0];
rz(-1.0245984938105013) q[0];
ry(-2.7680538432620803) q[1];
rz(2.948213928541048) q[1];
ry(3.0554284308421216) q[2];
rz(-1.8435202664349308) q[2];
ry(1.4219207289916085) q[3];
rz(0.8467958524069744) q[3];
ry(-1.1731763366062655) q[4];
rz(-1.6864753667493753) q[4];
ry(-1.9080543695432155) q[5];
rz(2.880948128848618) q[5];
ry(0.7948540140736959) q[6];
rz(2.812375950151007) q[6];
ry(2.4397395194293727) q[7];
rz(3.0291003443558573) q[7];
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
ry(-2.9835082424013932) q[0];
rz(-1.1286504010724545) q[0];
ry(2.5349864153063177) q[1];
rz(1.6995598894724329) q[1];
ry(1.086770523377423) q[2];
rz(2.110926146895971) q[2];
ry(2.2169746893889535) q[3];
rz(-3.05760981303808) q[3];
ry(-1.0144984240342954) q[4];
rz(2.697428143065947) q[4];
ry(-1.9802028517287824) q[5];
rz(-1.535329385552836) q[5];
ry(-2.7121464240227473) q[6];
rz(-0.6510581835753079) q[6];
ry(-2.0610723054667623) q[7];
rz(-0.6006725423200775) q[7];
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
ry(-0.7401988984075674) q[0];
rz(-0.21622840129217827) q[0];
ry(2.347119286718397) q[1];
rz(2.3822237754048246) q[1];
ry(-2.271699717607056) q[2];
rz(-1.253857559519887) q[2];
ry(1.7819337261956818) q[3];
rz(0.914219770704379) q[3];
ry(-1.4685855924994282) q[4];
rz(1.3351243335536118) q[4];
ry(-1.6710055464881872) q[5];
rz(2.99330340116832) q[5];
ry(-1.8674364229469973) q[6];
rz(0.4573575486343708) q[6];
ry(-2.6727881184382225) q[7];
rz(0.056275063211485225) q[7];
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
ry(-1.3614701205627773) q[0];
rz(-1.2361299589976897) q[0];
ry(-2.970549294321284) q[1];
rz(-3.1055287523095063) q[1];
ry(-2.296205849948775) q[2];
rz(1.8055155556501772) q[2];
ry(-2.8799937028344798) q[3];
rz(-2.1922724166085326) q[3];
ry(-0.4036300722518815) q[4];
rz(1.4655317438357407) q[4];
ry(-1.7006769451025605) q[5];
rz(1.0558663372071297) q[5];
ry(-2.2080849417230946) q[6];
rz(-2.3200401167347704) q[6];
ry(0.6492138148248936) q[7];
rz(-1.6475321690950937) q[7];
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
ry(-0.731080836133641) q[0];
rz(2.8727919525023666) q[0];
ry(0.7122327592033955) q[1];
rz(0.5849796044950564) q[1];
ry(-3.0390516378402834) q[2];
rz(-1.7090164002799781) q[2];
ry(2.646606080118688) q[3];
rz(-3.0731680209899017) q[3];
ry(1.3857723143396228) q[4];
rz(1.4423918221501566) q[4];
ry(-0.3671823683030864) q[5];
rz(0.23041665980238324) q[5];
ry(-1.8672452905091497) q[6];
rz(-2.0895498425936543) q[6];
ry(-1.4252693829404193) q[7];
rz(2.4436438425457183) q[7];
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
ry(-1.7000762815316133) q[0];
rz(0.29198961718476957) q[0];
ry(-2.220619864984412) q[1];
rz(-0.10842068987338166) q[1];
ry(-0.49309546892716044) q[2];
rz(-3.0233413081253873) q[2];
ry(2.399467311061667) q[3];
rz(-0.8060322757662101) q[3];
ry(0.6235949496346043) q[4];
rz(0.8206871422794308) q[4];
ry(-2.0191329028879528) q[5];
rz(-2.0908179086071743) q[5];
ry(-0.2916437978826338) q[6];
rz(0.5505338901473378) q[6];
ry(-1.8867821285569149) q[7];
rz(1.1537604069443548) q[7];
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
ry(-0.45737702220771226) q[0];
rz(-0.7967901705663376) q[0];
ry(-2.50084613183837) q[1];
rz(-1.1192737910257833) q[1];
ry(0.19234204131716393) q[2];
rz(2.6493229100386824) q[2];
ry(1.5985167018536126) q[3];
rz(1.494521625562597) q[3];
ry(2.209867327418361) q[4];
rz(1.6879636348441878) q[4];
ry(-0.7310284748640878) q[5];
rz(1.591701069085313) q[5];
ry(-0.4171087308234121) q[6];
rz(0.2818178943567707) q[6];
ry(-1.6106955255632327) q[7];
rz(-0.7766519818263804) q[7];
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
ry(1.0234358904020378) q[0];
rz(1.582284182512959) q[0];
ry(-0.5475081569404496) q[1];
rz(0.6872588610430399) q[1];
ry(-1.4709956278412042) q[2];
rz(0.3287207860541992) q[2];
ry(2.5537439071075436) q[3];
rz(1.9451319842471961) q[3];
ry(2.2392003376268867) q[4];
rz(0.6921169418413566) q[4];
ry(2.241744931976224) q[5];
rz(-1.5476582084801773) q[5];
ry(1.489515218834499) q[6];
rz(2.8580496290849333) q[6];
ry(-0.38618645657921413) q[7];
rz(-1.5662769942452495) q[7];
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
ry(1.6855640528752238) q[0];
rz(-0.6714434033386646) q[0];
ry(2.7728494626712186) q[1];
rz(-2.4402916283742564) q[1];
ry(-2.0221325666595487) q[2];
rz(-3.079775402686993) q[2];
ry(1.1922498985852048) q[3];
rz(1.8711827476310994) q[3];
ry(1.7135156836631282) q[4];
rz(2.1875916272511944) q[4];
ry(1.9344378473161783) q[5];
rz(-1.8733356722349903) q[5];
ry(-1.5488055671996606) q[6];
rz(-2.000244810930438) q[6];
ry(-1.1331470249421072) q[7];
rz(2.658691943250314) q[7];
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
ry(1.590848205931094) q[0];
rz(-2.6949803888459942) q[0];
ry(-1.3727557385565774) q[1];
rz(0.011972062182426005) q[1];
ry(0.9317370605867561) q[2];
rz(0.8693383803963711) q[2];
ry(-0.12538133389709283) q[3];
rz(1.4410003809061669) q[3];
ry(0.11413001193554415) q[4];
rz(0.9863866961593689) q[4];
ry(-0.3970215949153003) q[5];
rz(0.16751488069562287) q[5];
ry(-0.8134184996769046) q[6];
rz(0.7426004666357962) q[6];
ry(0.9469845276560207) q[7];
rz(2.7774011854874803) q[7];
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
ry(-0.4281942637534462) q[0];
rz(-0.5660699363212016) q[0];
ry(-1.608327805860048) q[1];
rz(-2.3762519122588373) q[1];
ry(1.2869960932077205) q[2];
rz(-1.0073528010911696) q[2];
ry(-0.824426515957688) q[3];
rz(-1.1958138148877735) q[3];
ry(1.8220392550165978) q[4];
rz(1.2902457809023924) q[4];
ry(-1.7769766271814396) q[5];
rz(-2.1455933055596357) q[5];
ry(-2.271978278109011) q[6];
rz(1.487183266539144) q[6];
ry(1.9058392137759725) q[7];
rz(-1.0778278914292674) q[7];
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
ry(2.624999020652765) q[0];
rz(1.4203080771344507) q[0];
ry(1.0921475440187072) q[1];
rz(0.7200831908321033) q[1];
ry(1.5589675091122335) q[2];
rz(1.0950315024933575) q[2];
ry(-2.572219309863515) q[3];
rz(-0.5585579494986729) q[3];
ry(-1.0310508265680358) q[4];
rz(1.1730963120903946) q[4];
ry(-1.1972467729966954) q[5];
rz(2.7266607116647963) q[5];
ry(-2.1365559866109676) q[6];
rz(1.768449805758376) q[6];
ry(-2.2914519878718496) q[7];
rz(-2.8895540999196196) q[7];
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
ry(1.659402586779545) q[0];
rz(0.11887655358981554) q[0];
ry(1.0584273742334225) q[1];
rz(-0.39680610513128917) q[1];
ry(2.88183974255724) q[2];
rz(-0.16643900106260734) q[2];
ry(-2.008688547625316) q[3];
rz(0.4726013258889168) q[3];
ry(-3.1086052141104887) q[4];
rz(0.05971616201242008) q[4];
ry(-2.4767139047282676) q[5];
rz(2.01378410632398) q[5];
ry(2.0947203646720114) q[6];
rz(-2.224795660252906) q[6];
ry(1.5549965485978454) q[7];
rz(0.7623987359065323) q[7];
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
ry(2.683389730087991) q[0];
rz(1.070643846282007) q[0];
ry(-2.100452347623947) q[1];
rz(-2.4274924695946196) q[1];
ry(-0.7235673158380509) q[2];
rz(-2.0662487314725873) q[2];
ry(-2.008804289116042) q[3];
rz(1.917382008013166) q[3];
ry(1.9386974308906197) q[4];
rz(-1.7604422180104005) q[4];
ry(0.942401832377179) q[5];
rz(1.7180464030601192) q[5];
ry(1.9613959234811613) q[6];
rz(0.6650345273891684) q[6];
ry(-1.7376454209332879) q[7];
rz(-0.45115119196348363) q[7];
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
ry(-2.782540425812732) q[0];
rz(1.4796633913780228) q[0];
ry(-2.7688384239157875) q[1];
rz(2.203713268112476) q[1];
ry(2.6135248168811) q[2];
rz(-2.0846861986075536) q[2];
ry(3.0373200714352717) q[3];
rz(-2.9846914548509837) q[3];
ry(1.6101218786853848) q[4];
rz(-1.8544409661612538) q[4];
ry(-1.94316592647849) q[5];
rz(2.5898428342942887) q[5];
ry(1.4819690543821737) q[6];
rz(-2.288819035458619) q[6];
ry(-1.3441415320157455) q[7];
rz(3.03940221995539) q[7];
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
ry(2.0559567938105303) q[0];
rz(-0.701996720019147) q[0];
ry(2.0854316222579854) q[1];
rz(-2.7511418340547515) q[1];
ry(-0.532983444660947) q[2];
rz(0.6943537953265615) q[2];
ry(2.1538338294027923) q[3];
rz(-0.3516512200366286) q[3];
ry(-1.8074125673813393) q[4];
rz(-2.9395394414611316) q[4];
ry(1.215239925673659) q[5];
rz(-0.7669705991860871) q[5];
ry(0.11389820481048396) q[6];
rz(-0.6443004080974957) q[6];
ry(-0.2634678151786636) q[7];
rz(-1.4310265768941215) q[7];
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
ry(0.7923868691443937) q[0];
rz(-1.082266810432082) q[0];
ry(1.7079609479468443) q[1];
rz(-2.3035612902013622) q[1];
ry(2.327074556697291) q[2];
rz(-1.7359121143370881) q[2];
ry(-2.434921643626869) q[3];
rz(0.2711729935319058) q[3];
ry(0.49783945684850966) q[4];
rz(1.5904426741223905) q[4];
ry(1.365188969971122) q[5];
rz(-2.753456375235548) q[5];
ry(-1.4775775323921332) q[6];
rz(1.849028545457999) q[6];
ry(-1.0415041502949827) q[7];
rz(-1.431282538455755) q[7];
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
ry(-1.5822933055671085) q[0];
rz(-2.391172376682248) q[0];
ry(-0.23782266176898137) q[1];
rz(-0.2334073574698614) q[1];
ry(-0.8159276257525407) q[2];
rz(1.590168686060614) q[2];
ry(-0.12432419703921127) q[3];
rz(-0.9672534760242869) q[3];
ry(-2.6384184229203265) q[4];
rz(-0.5967894017365277) q[4];
ry(-2.814585887313293) q[5];
rz(1.5074516860268163) q[5];
ry(-2.203600885712817) q[6];
rz(-2.7699161537360593) q[6];
ry(-2.537178128743656) q[7];
rz(2.8791929491715877) q[7];
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
ry(0.4332994837742872) q[0];
rz(0.11187896225094907) q[0];
ry(-0.10327081731134283) q[1];
rz(-0.15809565195773437) q[1];
ry(-1.5587627193612539) q[2];
rz(-0.8478044921846677) q[2];
ry(-1.4615734902362851) q[3];
rz(1.4087510986947354) q[3];
ry(-0.9550628001537875) q[4];
rz(1.7338406808541438) q[4];
ry(-2.734423617986444) q[5];
rz(2.8619672150532147) q[5];
ry(-2.304931795699447) q[6];
rz(2.9995635068848427) q[6];
ry(-1.7522073431282459) q[7];
rz(2.1653318731021702) q[7];
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
ry(-2.636794708000755) q[0];
rz(1.415538707689937) q[0];
ry(-0.41789841160402214) q[1];
rz(-2.186118022610458) q[1];
ry(2.445052821971022) q[2];
rz(2.33688788654121) q[2];
ry(2.3445452336243813) q[3];
rz(-0.6256663976500092) q[3];
ry(2.361104626330045) q[4];
rz(0.0037367341603466997) q[4];
ry(-3.082850042652063) q[5];
rz(2.9874196858202167) q[5];
ry(1.2195963045344547) q[6];
rz(0.3627345442479595) q[6];
ry(0.5077017274301472) q[7];
rz(-0.8027297563420008) q[7];
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
ry(2.5522032315967236) q[0];
rz(-0.8289891003498562) q[0];
ry(2.634260522621883) q[1];
rz(-2.990908241432324) q[1];
ry(-2.575237978426186) q[2];
rz(-1.5748753004694913) q[2];
ry(0.9678286554473026) q[3];
rz(-1.2956488380391058) q[3];
ry(1.5958884533549957) q[4];
rz(-0.7344998573012306) q[4];
ry(-1.8728074117848839) q[5];
rz(-1.0317164177496252) q[5];
ry(0.3701333487807182) q[6];
rz(-0.9584079930824508) q[6];
ry(0.525227231009964) q[7];
rz(-1.2496812470588634) q[7];
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
ry(0.9090278338040036) q[0];
rz(0.1838271218487886) q[0];
ry(2.8859823682500805) q[1];
rz(-0.9723223834842383) q[1];
ry(-2.744989613066588) q[2];
rz(0.09615473333706483) q[2];
ry(2.000679730804012) q[3];
rz(0.39762418140918315) q[3];
ry(0.20897617786145606) q[4];
rz(-1.7872530627903442) q[4];
ry(0.3569795947297512) q[5];
rz(2.4307208771050774) q[5];
ry(-1.6485469024862978) q[6];
rz(0.3055225752901117) q[6];
ry(-1.0887296034846088) q[7];
rz(-2.0334989756223445) q[7];
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
ry(-1.8621865859470488) q[0];
rz(-1.304432671367725) q[0];
ry(1.6720403002877904) q[1];
rz(-2.4630909961599854) q[1];
ry(-1.772870820527097) q[2];
rz(-2.770726080966526) q[2];
ry(0.7030453197118712) q[3];
rz(-2.292269191905293) q[3];
ry(0.27806110692986863) q[4];
rz(2.111204049932714) q[4];
ry(-2.8209274931642) q[5];
rz(0.02784366800316373) q[5];
ry(-1.2543374847965831) q[6];
rz(2.647368070714549) q[6];
ry(1.1499007319555568) q[7];
rz(3.1218989044339547) q[7];
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
ry(-0.5924800606273554) q[0];
rz(-1.6333687795884602) q[0];
ry(-2.200590247471643) q[1];
rz(-3.001712240154075) q[1];
ry(-0.4294274747167437) q[2];
rz(0.7915677394383104) q[2];
ry(1.5294845415245713) q[3];
rz(-0.39800523985112757) q[3];
ry(1.6853461126917273) q[4];
rz(1.3453842686490542) q[4];
ry(3.0694764158761427) q[5];
rz(0.5910481306017364) q[5];
ry(-1.4285245060529679) q[6];
rz(0.8441911198498687) q[6];
ry(-0.365361285981105) q[7];
rz(-2.9074178575366005) q[7];
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
ry(-1.6315275044813715) q[0];
rz(0.43150157608985307) q[0];
ry(-0.321209728819678) q[1];
rz(-2.7360554534160335) q[1];
ry(-0.8388785496413346) q[2];
rz(-1.612861923567136) q[2];
ry(1.2664692568002476) q[3];
rz(-0.7249956508714616) q[3];
ry(2.295803561625035) q[4];
rz(2.6017416068363124) q[4];
ry(-0.18766930696587547) q[5];
rz(2.272403565630416) q[5];
ry(-1.0086456804899022) q[6];
rz(-1.2016992858403093) q[6];
ry(0.2912725750537471) q[7];
rz(0.8243161698655443) q[7];
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
ry(0.5477882368121565) q[0];
rz(2.056101389824412) q[0];
ry(0.013475515735013887) q[1];
rz(-1.5156588561418898) q[1];
ry(-2.7038109720295758) q[2];
rz(2.4264251796197343) q[2];
ry(1.7039368061743525) q[3];
rz(0.9111465674246624) q[3];
ry(-2.386775134158344) q[4];
rz(2.2991678850224977) q[4];
ry(-1.9532679549901681) q[5];
rz(0.7270521286359486) q[5];
ry(-2.651948592345898) q[6];
rz(-0.631672554725576) q[6];
ry(-2.369122815650521) q[7];
rz(2.5739645306722463) q[7];