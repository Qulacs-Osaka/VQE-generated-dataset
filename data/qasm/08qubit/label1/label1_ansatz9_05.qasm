OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.903758741124798) q[0];
ry(0.26941317222868655) q[1];
cx q[0],q[1];
ry(-3.0501421755477898) q[0];
ry(-1.1265033638065076) q[1];
cx q[0],q[1];
ry(-0.7484468622896436) q[2];
ry(0.7832613847141737) q[3];
cx q[2],q[3];
ry(-2.6675165332970936) q[2];
ry(-1.9842606058128498) q[3];
cx q[2],q[3];
ry(-1.441469457389395) q[4];
ry(-0.34364450552368275) q[5];
cx q[4],q[5];
ry(-1.4010937288215892) q[4];
ry(2.8144814382061116) q[5];
cx q[4],q[5];
ry(-1.5224729059727258) q[6];
ry(0.05588933196419088) q[7];
cx q[6],q[7];
ry(1.5829046716091) q[6];
ry(-2.505265448863536) q[7];
cx q[6],q[7];
ry(-1.4400177822056497) q[0];
ry(-2.122521467642995) q[2];
cx q[0],q[2];
ry(2.63615133692023) q[0];
ry(2.569607966777863) q[2];
cx q[0],q[2];
ry(2.1572818601056816) q[2];
ry(-3.1063248821834226) q[4];
cx q[2],q[4];
ry(-1.520057193465087) q[2];
ry(1.307512336923284) q[4];
cx q[2],q[4];
ry(-1.3787928872828021) q[4];
ry(-1.2276601642783873) q[6];
cx q[4],q[6];
ry(1.724216154526885) q[4];
ry(-2.1973236284578865) q[6];
cx q[4],q[6];
ry(-1.235217630131248) q[1];
ry(0.24538276775099174) q[3];
cx q[1],q[3];
ry(-0.4037885396898955) q[1];
ry(0.9871498090980673) q[3];
cx q[1],q[3];
ry(-0.2664592581259564) q[3];
ry(0.0964206163825878) q[5];
cx q[3],q[5];
ry(-0.2727319893594631) q[3];
ry(2.981877110901722) q[5];
cx q[3],q[5];
ry(-3.1062139213668063) q[5];
ry(-2.4564637200595234) q[7];
cx q[5],q[7];
ry(1.697619355332366) q[5];
ry(-0.1304674837833756) q[7];
cx q[5],q[7];
ry(1.2889446465743075) q[0];
ry(-2.551493087778418) q[3];
cx q[0],q[3];
ry(1.2350133531526157) q[0];
ry(1.8582069709814686) q[3];
cx q[0],q[3];
ry(-0.05435674208124386) q[1];
ry(2.7925911334096436) q[2];
cx q[1],q[2];
ry(-1.4945510096323227) q[1];
ry(-1.4580964509080416) q[2];
cx q[1],q[2];
ry(0.2677772578263493) q[2];
ry(2.5933629447399373) q[5];
cx q[2],q[5];
ry(1.3670085884305847) q[2];
ry(-2.163549441082899) q[5];
cx q[2],q[5];
ry(2.26644210186729) q[3];
ry(0.9006311610637526) q[4];
cx q[3],q[4];
ry(0.6464286505935779) q[3];
ry(0.44012956213262766) q[4];
cx q[3],q[4];
ry(-1.2397245469777622) q[4];
ry(-3.051446808213257) q[7];
cx q[4],q[7];
ry(-2.373246830608898) q[4];
ry(-1.6603003312476288) q[7];
cx q[4],q[7];
ry(-3.0468450908801654) q[5];
ry(-0.8986318714736159) q[6];
cx q[5],q[6];
ry(2.677343860970575) q[5];
ry(1.0084355359711399) q[6];
cx q[5],q[6];
ry(1.502488345517816) q[0];
ry(1.9508416737664935) q[1];
cx q[0],q[1];
ry(2.581705886279271) q[0];
ry(-2.9802894482596884) q[1];
cx q[0],q[1];
ry(1.7035064068654808) q[2];
ry(0.10268854831850228) q[3];
cx q[2],q[3];
ry(2.259748989955222) q[2];
ry(-0.4115183656904265) q[3];
cx q[2],q[3];
ry(-1.7370888073372477) q[4];
ry(2.172038053694758) q[5];
cx q[4],q[5];
ry(-3.091164617708669) q[4];
ry(1.1062785010213032) q[5];
cx q[4],q[5];
ry(-2.373502344228268) q[6];
ry(-2.1948761733677897) q[7];
cx q[6],q[7];
ry(1.1754851331181193) q[6];
ry(0.7458576663068479) q[7];
cx q[6],q[7];
ry(1.7955484744759458) q[0];
ry(-0.5200572325298358) q[2];
cx q[0],q[2];
ry(1.1265006769263861) q[0];
ry(-1.589215518897228) q[2];
cx q[0],q[2];
ry(-1.3665417291074498) q[2];
ry(-2.3965951799754173) q[4];
cx q[2],q[4];
ry(-0.9022882423916423) q[2];
ry(1.4969040987195148) q[4];
cx q[2],q[4];
ry(-0.6861170857303296) q[4];
ry(-0.8968033150071829) q[6];
cx q[4],q[6];
ry(2.09482408238191) q[4];
ry(-1.3516616663406733) q[6];
cx q[4],q[6];
ry(1.3920571843118825) q[1];
ry(2.17832008508892) q[3];
cx q[1],q[3];
ry(0.5892842794874804) q[1];
ry(-2.0292944766601115) q[3];
cx q[1],q[3];
ry(2.6042477686902403) q[3];
ry(-0.17690243970725472) q[5];
cx q[3],q[5];
ry(0.11894379389936616) q[3];
ry(0.34031595377062684) q[5];
cx q[3],q[5];
ry(0.36591736273920666) q[5];
ry(-1.5487693717471764) q[7];
cx q[5],q[7];
ry(-1.8613565272688737) q[5];
ry(1.5375875854901508) q[7];
cx q[5],q[7];
ry(2.140650043797023) q[0];
ry(0.4553791386856224) q[3];
cx q[0],q[3];
ry(2.6837971246478527) q[0];
ry(-0.9689787055531445) q[3];
cx q[0],q[3];
ry(-1.0926259353680798) q[1];
ry(-0.980362712743216) q[2];
cx q[1],q[2];
ry(1.7103888953675688) q[1];
ry(2.5973447371418525) q[2];
cx q[1],q[2];
ry(0.3698796858119371) q[2];
ry(-0.9774729366803072) q[5];
cx q[2],q[5];
ry(-0.07517047486845652) q[2];
ry(0.16042597284255058) q[5];
cx q[2],q[5];
ry(-0.7762471708460641) q[3];
ry(-2.9945484361685692) q[4];
cx q[3],q[4];
ry(2.51348552460069) q[3];
ry(-1.2124742368767476) q[4];
cx q[3],q[4];
ry(2.9397404661213806) q[4];
ry(-1.1661940267768003) q[7];
cx q[4],q[7];
ry(2.896946875571039) q[4];
ry(1.2520279749135004) q[7];
cx q[4],q[7];
ry(0.4134126341027712) q[5];
ry(-1.192767039906351) q[6];
cx q[5],q[6];
ry(0.9523072113233872) q[5];
ry(1.367196443747357) q[6];
cx q[5],q[6];
ry(2.9232381184254232) q[0];
ry(0.014687160014612895) q[1];
cx q[0],q[1];
ry(-1.8645643881290397) q[0];
ry(2.560381408972311) q[1];
cx q[0],q[1];
ry(-1.8566002018581456) q[2];
ry(0.6744366243918449) q[3];
cx q[2],q[3];
ry(-3.0805668658386747) q[2];
ry(-1.477416079906739) q[3];
cx q[2],q[3];
ry(-2.747744993174401) q[4];
ry(-2.191995395437234) q[5];
cx q[4],q[5];
ry(-0.5510962285356078) q[4];
ry(2.3293563027732924) q[5];
cx q[4],q[5];
ry(0.7370926129570057) q[6];
ry(2.9677882212734796) q[7];
cx q[6],q[7];
ry(-0.0025545652515752693) q[6];
ry(-1.8121822598147173) q[7];
cx q[6],q[7];
ry(-1.1826453103949872) q[0];
ry(-2.1545363486479134) q[2];
cx q[0],q[2];
ry(2.665389510110884) q[0];
ry(-2.796835406319449) q[2];
cx q[0],q[2];
ry(-1.6248385274475685) q[2];
ry(2.053748031998591) q[4];
cx q[2],q[4];
ry(0.4974191674652749) q[2];
ry(-1.7431945170228786) q[4];
cx q[2],q[4];
ry(2.9577068575353485) q[4];
ry(-2.9807949901259265) q[6];
cx q[4],q[6];
ry(1.7168981273998682) q[4];
ry(-2.6085745155768185) q[6];
cx q[4],q[6];
ry(-1.7706866319034433) q[1];
ry(-1.9171924329159513) q[3];
cx q[1],q[3];
ry(1.977342631743408) q[1];
ry(0.8077166043371599) q[3];
cx q[1],q[3];
ry(-2.0589595690516616) q[3];
ry(-1.2942452102926305) q[5];
cx q[3],q[5];
ry(0.7568521712581004) q[3];
ry(-0.8670258821186541) q[5];
cx q[3],q[5];
ry(1.8753269794289995) q[5];
ry(1.0885052338771886) q[7];
cx q[5],q[7];
ry(-0.4937378900156823) q[5];
ry(-1.238903068264156) q[7];
cx q[5],q[7];
ry(-1.3927482016110948) q[0];
ry(-2.0742327771783913) q[3];
cx q[0],q[3];
ry(-1.4361137449854782) q[0];
ry(0.8265853716315112) q[3];
cx q[0],q[3];
ry(-0.018400159839608854) q[1];
ry(0.6059911383837896) q[2];
cx q[1],q[2];
ry(1.9805015308276146) q[1];
ry(-2.89154884145597) q[2];
cx q[1],q[2];
ry(-2.7575438643647474) q[2];
ry(2.7883296867729666) q[5];
cx q[2],q[5];
ry(-0.36986169237289346) q[2];
ry(-2.128817829010103) q[5];
cx q[2],q[5];
ry(-1.1797352024223762) q[3];
ry(-2.478134657678519) q[4];
cx q[3],q[4];
ry(-0.8638272259250689) q[3];
ry(-2.74427692811474) q[4];
cx q[3],q[4];
ry(-1.4424578779238066) q[4];
ry(0.6252764109465714) q[7];
cx q[4],q[7];
ry(1.9922131361200974) q[4];
ry(-1.9010719174477098) q[7];
cx q[4],q[7];
ry(2.802910160202389) q[5];
ry(-1.0340187512046013) q[6];
cx q[5],q[6];
ry(0.8302218697103954) q[5];
ry(1.9486013214532003) q[6];
cx q[5],q[6];
ry(-0.8998694842304994) q[0];
ry(-1.403539311644651) q[1];
cx q[0],q[1];
ry(-1.539649853593203) q[0];
ry(-2.329853517834888) q[1];
cx q[0],q[1];
ry(-0.2601403026687583) q[2];
ry(-1.8216456201306324) q[3];
cx q[2],q[3];
ry(2.9557234301281428) q[2];
ry(-0.8709920953389956) q[3];
cx q[2],q[3];
ry(0.039979657823355635) q[4];
ry(0.7860132608884369) q[5];
cx q[4],q[5];
ry(2.8052368450275567) q[4];
ry(0.6140292866125749) q[5];
cx q[4],q[5];
ry(2.973658456486994) q[6];
ry(-0.889783799798324) q[7];
cx q[6],q[7];
ry(1.9149217870176525) q[6];
ry(1.2612513784639825) q[7];
cx q[6],q[7];
ry(-0.12001631824840966) q[0];
ry(-1.4788013914331464) q[2];
cx q[0],q[2];
ry(0.9602800097752955) q[0];
ry(-1.4310753670369198) q[2];
cx q[0],q[2];
ry(-1.8659896597272203) q[2];
ry(-2.7633045667004823) q[4];
cx q[2],q[4];
ry(-2.0954191849194412) q[2];
ry(1.1475036995991257) q[4];
cx q[2],q[4];
ry(-2.0160144336977135) q[4];
ry(0.4439244464997785) q[6];
cx q[4],q[6];
ry(-2.0015600871104846) q[4];
ry(2.0927744009879365) q[6];
cx q[4],q[6];
ry(-0.5192408077882478) q[1];
ry(-1.362026016575986) q[3];
cx q[1],q[3];
ry(0.667864212565001) q[1];
ry(3.0851338941974613) q[3];
cx q[1],q[3];
ry(0.2951305550095382) q[3];
ry(0.9369801715090706) q[5];
cx q[3],q[5];
ry(3.0915409028916505) q[3];
ry(0.10351836301352993) q[5];
cx q[3],q[5];
ry(-2.390173161743972) q[5];
ry(-1.161118101995274) q[7];
cx q[5],q[7];
ry(0.759522613086813) q[5];
ry(-0.7922635630572135) q[7];
cx q[5],q[7];
ry(-2.230285503900177) q[0];
ry(-3.1064988740014976) q[3];
cx q[0],q[3];
ry(-2.94022111607187) q[0];
ry(1.7102822458245985) q[3];
cx q[0],q[3];
ry(-2.04635630310497) q[1];
ry(0.4610963837550031) q[2];
cx q[1],q[2];
ry(-2.591150074780324) q[1];
ry(1.225764375007865) q[2];
cx q[1],q[2];
ry(2.055683506145935) q[2];
ry(1.1205893082573963) q[5];
cx q[2],q[5];
ry(-0.225213556161777) q[2];
ry(0.49249086343231885) q[5];
cx q[2],q[5];
ry(-0.7213140066665629) q[3];
ry(-1.1630332085558126) q[4];
cx q[3],q[4];
ry(1.760713700244887) q[3];
ry(-0.23341969813943186) q[4];
cx q[3],q[4];
ry(-0.5495677227006919) q[4];
ry(0.04454569801043253) q[7];
cx q[4],q[7];
ry(2.2715784997050084) q[4];
ry(-0.6200334765449407) q[7];
cx q[4],q[7];
ry(2.5881890505001057) q[5];
ry(0.4067469911430893) q[6];
cx q[5],q[6];
ry(0.6154044912167986) q[5];
ry(2.9103428305351424) q[6];
cx q[5],q[6];
ry(1.2424347860395848) q[0];
ry(0.9397624498857686) q[1];
cx q[0],q[1];
ry(2.3682574408562105) q[0];
ry(-1.5873314938554715) q[1];
cx q[0],q[1];
ry(1.1080412661241965) q[2];
ry(-3.0999474788489536) q[3];
cx q[2],q[3];
ry(-1.8306328562605128) q[2];
ry(-0.041348751254624894) q[3];
cx q[2],q[3];
ry(-1.0070653907954359) q[4];
ry(-0.006746268142581652) q[5];
cx q[4],q[5];
ry(2.531410077303912) q[4];
ry(-1.8300236571903046) q[5];
cx q[4],q[5];
ry(-0.18047501384607958) q[6];
ry(1.3105846652747495) q[7];
cx q[6],q[7];
ry(-0.27020871404837177) q[6];
ry(0.2776545874077003) q[7];
cx q[6],q[7];
ry(-1.9740429750420834) q[0];
ry(-1.267791512581833) q[2];
cx q[0],q[2];
ry(2.191691010484474) q[0];
ry(-0.8642336398090151) q[2];
cx q[0],q[2];
ry(2.240485108101197) q[2];
ry(0.8800334983293867) q[4];
cx q[2],q[4];
ry(-2.6621786094469098) q[2];
ry(2.0093618133693782) q[4];
cx q[2],q[4];
ry(0.3815900924892437) q[4];
ry(0.43610196267750967) q[6];
cx q[4],q[6];
ry(1.1275895585227342) q[4];
ry(-1.4517550066636922) q[6];
cx q[4],q[6];
ry(-1.6897713535050816) q[1];
ry(0.7792590090128307) q[3];
cx q[1],q[3];
ry(1.7287455252047712) q[1];
ry(2.7688204574473545) q[3];
cx q[1],q[3];
ry(-3.1163888575915495) q[3];
ry(2.155319567121205) q[5];
cx q[3],q[5];
ry(-2.901251543677648) q[3];
ry(2.7923995423800743) q[5];
cx q[3],q[5];
ry(-0.8988526207510653) q[5];
ry(-2.8852018720012986) q[7];
cx q[5],q[7];
ry(2.24797638398931) q[5];
ry(-2.09322385558728) q[7];
cx q[5],q[7];
ry(-1.0279866819449328) q[0];
ry(-1.290660722430978) q[3];
cx q[0],q[3];
ry(-1.1803279730019858) q[0];
ry(0.6629625357319151) q[3];
cx q[0],q[3];
ry(-2.9945521164695026) q[1];
ry(-2.799120601234117) q[2];
cx q[1],q[2];
ry(-2.521863768758643) q[1];
ry(1.812906739120768) q[2];
cx q[1],q[2];
ry(-1.3748662314929092) q[2];
ry(0.8818393514429079) q[5];
cx q[2],q[5];
ry(2.317846920233255) q[2];
ry(-2.8958185154506713) q[5];
cx q[2],q[5];
ry(1.0036773149265255) q[3];
ry(-2.307515948633621) q[4];
cx q[3],q[4];
ry(-1.245249779506028) q[3];
ry(1.7264650802296786) q[4];
cx q[3],q[4];
ry(-1.590977919769378) q[4];
ry(1.2320443899642053) q[7];
cx q[4],q[7];
ry(-3.106471144026808) q[4];
ry(0.6733429913816167) q[7];
cx q[4],q[7];
ry(-1.994712578566887) q[5];
ry(0.7372683534192062) q[6];
cx q[5],q[6];
ry(2.0094488579852983) q[5];
ry(-2.6105238455513535) q[6];
cx q[5],q[6];
ry(-0.4463437301441502) q[0];
ry(-1.8426683368769998) q[1];
cx q[0],q[1];
ry(-0.13747400272344645) q[0];
ry(-1.9701189627337798) q[1];
cx q[0],q[1];
ry(-1.808813969062833) q[2];
ry(-0.6246834770311631) q[3];
cx q[2],q[3];
ry(3.0667229299357954) q[2];
ry(1.1214639829116413) q[3];
cx q[2],q[3];
ry(-2.9020948542631952) q[4];
ry(-1.9647366539890678) q[5];
cx q[4],q[5];
ry(-0.48432682432639507) q[4];
ry(-1.5141850077968353) q[5];
cx q[4],q[5];
ry(-0.7766788190294297) q[6];
ry(2.8654574431078683) q[7];
cx q[6],q[7];
ry(-2.2842106446878097) q[6];
ry(-0.7055836239717115) q[7];
cx q[6],q[7];
ry(-3.0400591503968153) q[0];
ry(-0.6541626106949918) q[2];
cx q[0],q[2];
ry(1.4437136249041165) q[0];
ry(-3.049224308058104) q[2];
cx q[0],q[2];
ry(-2.3388747145644033) q[2];
ry(2.2311325797070234) q[4];
cx q[2],q[4];
ry(-1.5970505531712318) q[2];
ry(0.37554246293020377) q[4];
cx q[2],q[4];
ry(2.798022755135706) q[4];
ry(-1.893342168232249) q[6];
cx q[4],q[6];
ry(1.5399666485565815) q[4];
ry(-2.7178705854268146) q[6];
cx q[4],q[6];
ry(-0.20604244341480982) q[1];
ry(0.9522318974138901) q[3];
cx q[1],q[3];
ry(-0.9498981187726435) q[1];
ry(-0.39791110209214686) q[3];
cx q[1],q[3];
ry(-2.581827225620659) q[3];
ry(-1.5891747751327923) q[5];
cx q[3],q[5];
ry(1.440572916202992) q[3];
ry(0.19022699672755947) q[5];
cx q[3],q[5];
ry(-2.7412459525884385) q[5];
ry(0.3128183218624594) q[7];
cx q[5],q[7];
ry(-1.3738583871124543) q[5];
ry(-3.0661555946436057) q[7];
cx q[5],q[7];
ry(1.4776095966376293) q[0];
ry(2.804791501457658) q[3];
cx q[0],q[3];
ry(2.27201592643602) q[0];
ry(-0.5242720967123297) q[3];
cx q[0],q[3];
ry(0.0025462785926568606) q[1];
ry(1.7762884631192186) q[2];
cx q[1],q[2];
ry(0.7104302771604969) q[1];
ry(1.8157177907374145) q[2];
cx q[1],q[2];
ry(-2.4377252249835695) q[2];
ry(-2.5478268731562865) q[5];
cx q[2],q[5];
ry(2.7075056209507307) q[2];
ry(-0.2702832298664095) q[5];
cx q[2],q[5];
ry(-1.8759982699931586) q[3];
ry(-2.8615716820886714) q[4];
cx q[3],q[4];
ry(-0.6213792065559645) q[3];
ry(0.5647302212003025) q[4];
cx q[3],q[4];
ry(-2.236355036711093) q[4];
ry(-0.21964942600031256) q[7];
cx q[4],q[7];
ry(-1.8297667790422727) q[4];
ry(-1.7929729198390678) q[7];
cx q[4],q[7];
ry(-1.4678938370291146) q[5];
ry(0.9751497754509336) q[6];
cx q[5],q[6];
ry(1.5674099214645716) q[5];
ry(0.585344793729428) q[6];
cx q[5],q[6];
ry(-0.18309517489590554) q[0];
ry(1.1088115837384913) q[1];
cx q[0],q[1];
ry(1.4822640030274838) q[0];
ry(-2.1966748150156237) q[1];
cx q[0],q[1];
ry(-1.4801566263452361) q[2];
ry(-2.2827737514866575) q[3];
cx q[2],q[3];
ry(0.8379123776257537) q[2];
ry(-0.49986307822091525) q[3];
cx q[2],q[3];
ry(-2.6187718050979436) q[4];
ry(-0.41425362641116337) q[5];
cx q[4],q[5];
ry(-1.3052304624432187) q[4];
ry(1.9075520415500797) q[5];
cx q[4],q[5];
ry(-0.8987343292877279) q[6];
ry(1.867100448784213) q[7];
cx q[6],q[7];
ry(2.4320357988203294) q[6];
ry(-1.7079875797147024) q[7];
cx q[6],q[7];
ry(-0.790146261700738) q[0];
ry(2.69872184000475) q[2];
cx q[0],q[2];
ry(1.4986203974063188) q[0];
ry(-1.8577894432351996) q[2];
cx q[0],q[2];
ry(2.9478527295202968) q[2];
ry(0.5326289050368389) q[4];
cx q[2],q[4];
ry(1.0397617194862485) q[2];
ry(1.8209114954985626) q[4];
cx q[2],q[4];
ry(1.0126583689719704) q[4];
ry(-2.070720775514328) q[6];
cx q[4],q[6];
ry(-2.232924745114003) q[4];
ry(-2.046262776796218) q[6];
cx q[4],q[6];
ry(2.9181156815625733) q[1];
ry(-1.4610135499663377) q[3];
cx q[1],q[3];
ry(-2.0261768081521776) q[1];
ry(1.567699247979344) q[3];
cx q[1],q[3];
ry(0.9846614500311707) q[3];
ry(-3.068897817212269) q[5];
cx q[3],q[5];
ry(3.1290416140549757) q[3];
ry(-0.9122690734980886) q[5];
cx q[3],q[5];
ry(-2.0841274772431433) q[5];
ry(2.0950613643682416) q[7];
cx q[5],q[7];
ry(2.5431908215690733) q[5];
ry(-1.2686875841243705) q[7];
cx q[5],q[7];
ry(-0.03140212011093002) q[0];
ry(2.685969685186505) q[3];
cx q[0],q[3];
ry(2.7054561077562744) q[0];
ry(0.8075304331020491) q[3];
cx q[0],q[3];
ry(1.1934139658919714) q[1];
ry(-0.8026099216802846) q[2];
cx q[1],q[2];
ry(2.6342392264971886) q[1];
ry(-0.15902729139196306) q[2];
cx q[1],q[2];
ry(-0.642006639421513) q[2];
ry(-2.834694527790051) q[5];
cx q[2],q[5];
ry(2.731795101348513) q[2];
ry(-0.2984239600554511) q[5];
cx q[2],q[5];
ry(-0.14661760236621915) q[3];
ry(-2.24478147599525) q[4];
cx q[3],q[4];
ry(2.1156550270267855) q[3];
ry(-1.941527126917702) q[4];
cx q[3],q[4];
ry(-0.5092725132969198) q[4];
ry(-2.217949261262574) q[7];
cx q[4],q[7];
ry(-2.320827102703118) q[4];
ry(-2.5192541690131396) q[7];
cx q[4],q[7];
ry(0.28233620903383816) q[5];
ry(2.945300178928686) q[6];
cx q[5],q[6];
ry(-2.9153773286144933) q[5];
ry(-0.9393510085328748) q[6];
cx q[5],q[6];
ry(-0.5633867009198745) q[0];
ry(-1.1136387790201319) q[1];
cx q[0],q[1];
ry(0.4720059403719333) q[0];
ry(-2.218352925073063) q[1];
cx q[0],q[1];
ry(-0.4632035237582808) q[2];
ry(1.8956779500222711) q[3];
cx q[2],q[3];
ry(1.7136657259635273) q[2];
ry(-0.8677660919038341) q[3];
cx q[2],q[3];
ry(-2.9550585654770276) q[4];
ry(0.16835750297607266) q[5];
cx q[4],q[5];
ry(-1.1406788259041614) q[4];
ry(-2.7926982859761917) q[5];
cx q[4],q[5];
ry(0.2924361127433208) q[6];
ry(-3.0293116143058723) q[7];
cx q[6],q[7];
ry(-0.8523171564396606) q[6];
ry(-3.0336227173841817) q[7];
cx q[6],q[7];
ry(1.496977401330392) q[0];
ry(1.591864803430112) q[2];
cx q[0],q[2];
ry(-0.009466833368176886) q[0];
ry(-2.4332568487020785) q[2];
cx q[0],q[2];
ry(-1.7478802664110435) q[2];
ry(-1.6522255005419848) q[4];
cx q[2],q[4];
ry(3.123091902016211) q[2];
ry(1.461874213476896) q[4];
cx q[2],q[4];
ry(2.8978689381570266) q[4];
ry(1.084731064668155) q[6];
cx q[4],q[6];
ry(-1.5019345404598479) q[4];
ry(1.5923891704645596) q[6];
cx q[4],q[6];
ry(-1.2117111837241366) q[1];
ry(2.654116102540726) q[3];
cx q[1],q[3];
ry(1.1012316924861278) q[1];
ry(0.45756878358071873) q[3];
cx q[1],q[3];
ry(-1.14390021958457) q[3];
ry(3.0148552265176067) q[5];
cx q[3],q[5];
ry(-1.4565753979180658) q[3];
ry(1.8528812185305885) q[5];
cx q[3],q[5];
ry(-1.578858269766194) q[5];
ry(0.7732806868691169) q[7];
cx q[5],q[7];
ry(-2.6025746123052156) q[5];
ry(2.715894928398011) q[7];
cx q[5],q[7];
ry(2.9173797172875267) q[0];
ry(0.8480332691201742) q[3];
cx q[0],q[3];
ry(-1.6855156585243412) q[0];
ry(0.7216739223950235) q[3];
cx q[0],q[3];
ry(2.5752650130982193) q[1];
ry(2.6284918858552264) q[2];
cx q[1],q[2];
ry(-1.6103564671942758) q[1];
ry(2.2595335389550972) q[2];
cx q[1],q[2];
ry(1.9007419623288522) q[2];
ry(2.7224881237949288) q[5];
cx q[2],q[5];
ry(-1.3367316116507046) q[2];
ry(0.22475367772203014) q[5];
cx q[2],q[5];
ry(0.9813112667999251) q[3];
ry(-1.738042301720259) q[4];
cx q[3],q[4];
ry(-1.2932796375444955) q[3];
ry(1.8307850350532366) q[4];
cx q[3],q[4];
ry(2.7480342384454657) q[4];
ry(-2.4869786921169044) q[7];
cx q[4],q[7];
ry(0.9369080076026127) q[4];
ry(-1.0907699441086152) q[7];
cx q[4],q[7];
ry(1.806101665864837) q[5];
ry(0.40763648494238414) q[6];
cx q[5],q[6];
ry(-2.2270179863742463) q[5];
ry(-0.09078116949905812) q[6];
cx q[5],q[6];
ry(2.413081802269609) q[0];
ry(0.3312354659835268) q[1];
ry(1.5668949355709103) q[2];
ry(0.3795258160072601) q[3];
ry(-0.8847057283535174) q[4];
ry(-2.392832081285646) q[5];
ry(-1.7113199157217056) q[6];
ry(-2.998879867940339) q[7];