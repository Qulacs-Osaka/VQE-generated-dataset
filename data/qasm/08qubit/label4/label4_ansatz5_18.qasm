OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.6123645230451471) q[0];
ry(2.321365429636081) q[1];
cx q[0],q[1];
ry(0.45611330487089136) q[0];
ry(1.8383282209040717) q[1];
cx q[0],q[1];
ry(2.790604198609392) q[2];
ry(2.9829851480317777) q[3];
cx q[2],q[3];
ry(-1.6629223801929829) q[2];
ry(-1.6403083322905774) q[3];
cx q[2],q[3];
ry(2.776170092797422) q[4];
ry(1.4395230205261471) q[5];
cx q[4],q[5];
ry(-2.798790396451857) q[4];
ry(3.040897087072862) q[5];
cx q[4],q[5];
ry(2.8456143830119935) q[6];
ry(0.8066386620280781) q[7];
cx q[6],q[7];
ry(-1.5624825411987455) q[6];
ry(-2.2279546585926786) q[7];
cx q[6],q[7];
ry(-1.0162473240992282) q[1];
ry(2.9556050850453377) q[2];
cx q[1],q[2];
ry(1.4470369155323957) q[1];
ry(2.136643873628154) q[2];
cx q[1],q[2];
ry(2.717696666983295) q[3];
ry(-2.1637359876941114) q[4];
cx q[3],q[4];
ry(2.4482041403885226) q[3];
ry(2.2057609833864817) q[4];
cx q[3],q[4];
ry(0.02747189855123904) q[5];
ry(2.8022665354229566) q[6];
cx q[5],q[6];
ry(-0.8450759975422004) q[5];
ry(0.8175569082959472) q[6];
cx q[5],q[6];
ry(-1.0005435542637144) q[0];
ry(-0.1955173144392056) q[1];
cx q[0],q[1];
ry(0.3067459298003366) q[0];
ry(1.7764337020450245) q[1];
cx q[0],q[1];
ry(-3.1364253176625243) q[2];
ry(-0.5778827582591415) q[3];
cx q[2],q[3];
ry(-0.020373448257834603) q[2];
ry(-1.58492752390252) q[3];
cx q[2],q[3];
ry(-2.1622729980675386) q[4];
ry(2.768447581005291) q[5];
cx q[4],q[5];
ry(2.420727614699345) q[4];
ry(-2.3011915292644116) q[5];
cx q[4],q[5];
ry(1.0532704878929058) q[6];
ry(1.9156448102691563) q[7];
cx q[6],q[7];
ry(-2.508400105839715) q[6];
ry(-3.0821284050280062) q[7];
cx q[6],q[7];
ry(-1.8208369284466217) q[1];
ry(-1.2819943403469702) q[2];
cx q[1],q[2];
ry(-2.754429758453098) q[1];
ry(-3.1104716767938037) q[2];
cx q[1],q[2];
ry(2.321758214203341) q[3];
ry(0.027605533794118182) q[4];
cx q[3],q[4];
ry(1.6851233407458828) q[3];
ry(-2.817663904864663) q[4];
cx q[3],q[4];
ry(-1.9272527139601245) q[5];
ry(2.8803869867369385) q[6];
cx q[5],q[6];
ry(-1.3522694695480988) q[5];
ry(-0.8162213810938843) q[6];
cx q[5],q[6];
ry(0.5321267476075765) q[0];
ry(0.6350771528940451) q[1];
cx q[0],q[1];
ry(0.1511382764831639) q[0];
ry(-0.9999549658565102) q[1];
cx q[0],q[1];
ry(-1.3839238362141213) q[2];
ry(-1.223573116891622) q[3];
cx q[2],q[3];
ry(0.6348404676105778) q[2];
ry(-2.902237725648143) q[3];
cx q[2],q[3];
ry(-0.08333692344536825) q[4];
ry(1.7687349566777646) q[5];
cx q[4],q[5];
ry(1.1525503758379119) q[4];
ry(0.9357185617173429) q[5];
cx q[4],q[5];
ry(-1.858941393759232) q[6];
ry(0.3665162445486638) q[7];
cx q[6],q[7];
ry(2.0851598787068846) q[6];
ry(-1.577811309883499) q[7];
cx q[6],q[7];
ry(-1.0667869547482656) q[1];
ry(-1.914410035363272) q[2];
cx q[1],q[2];
ry(1.4788674192157971) q[1];
ry(1.7004136107809185) q[2];
cx q[1],q[2];
ry(2.794128445082673) q[3];
ry(3.0679610693369996) q[4];
cx q[3],q[4];
ry(-2.789691144741624) q[3];
ry(-3.0508253367411897) q[4];
cx q[3],q[4];
ry(-2.191322467503087) q[5];
ry(-0.10315102856310521) q[6];
cx q[5],q[6];
ry(-1.5683027757347543) q[5];
ry(0.15951701660019602) q[6];
cx q[5],q[6];
ry(1.2228324136049444) q[0];
ry(2.17131934393673) q[1];
cx q[0],q[1];
ry(0.14895719879195068) q[0];
ry(2.102041050691504) q[1];
cx q[0],q[1];
ry(2.949208010246479) q[2];
ry(-2.680043123385968) q[3];
cx q[2],q[3];
ry(-2.615722974933906) q[2];
ry(2.5146063502162828) q[3];
cx q[2],q[3];
ry(1.5456374660442305) q[4];
ry(0.2421935096591499) q[5];
cx q[4],q[5];
ry(1.055148902007982) q[4];
ry(-2.029001497781598) q[5];
cx q[4],q[5];
ry(1.5830491199019983) q[6];
ry(1.2674071636832833) q[7];
cx q[6],q[7];
ry(-0.7703048690123734) q[6];
ry(-0.324913385157541) q[7];
cx q[6],q[7];
ry(-2.9077151125769096) q[1];
ry(-2.40145161089077) q[2];
cx q[1],q[2];
ry(-0.12596654327037093) q[1];
ry(2.7160224518354483) q[2];
cx q[1],q[2];
ry(1.7269648757115308) q[3];
ry(0.7558908366163926) q[4];
cx q[3],q[4];
ry(1.4467529359293652) q[3];
ry(-0.1814324233386966) q[4];
cx q[3],q[4];
ry(-1.2394190337784456) q[5];
ry(-1.1938482572603322) q[6];
cx q[5],q[6];
ry(-2.2214096003037076) q[5];
ry(2.3087197426634227) q[6];
cx q[5],q[6];
ry(2.638147050703972) q[0];
ry(0.506266992285743) q[1];
cx q[0],q[1];
ry(2.7829528600830415) q[0];
ry(-1.0210524467987856) q[1];
cx q[0],q[1];
ry(-0.6657622858180928) q[2];
ry(0.7510282025476264) q[3];
cx q[2],q[3];
ry(1.6205897908882) q[2];
ry(1.9251429604869532) q[3];
cx q[2],q[3];
ry(2.573228636074142) q[4];
ry(-0.7855284267201198) q[5];
cx q[4],q[5];
ry(0.8987283719401802) q[4];
ry(-1.4082280372797804) q[5];
cx q[4],q[5];
ry(2.8439267748472874) q[6];
ry(0.09805413977812183) q[7];
cx q[6],q[7];
ry(1.733435395286482) q[6];
ry(-1.605879144877346) q[7];
cx q[6],q[7];
ry(-0.5562446615671126) q[1];
ry(1.06363591064608) q[2];
cx q[1],q[2];
ry(0.8323200785600723) q[1];
ry(-3.0276005217436617) q[2];
cx q[1],q[2];
ry(-0.4954783284573665) q[3];
ry(0.6644537061445859) q[4];
cx q[3],q[4];
ry(2.860968477085723) q[3];
ry(3.1215300804355657) q[4];
cx q[3],q[4];
ry(2.402432810828324) q[5];
ry(1.8594654808255857) q[6];
cx q[5],q[6];
ry(2.661738640257397) q[5];
ry(0.8217426821358328) q[6];
cx q[5],q[6];
ry(-1.537162419179638) q[0];
ry(-2.7957349282427484) q[1];
cx q[0],q[1];
ry(1.7463772342695583) q[0];
ry(0.5373308866748077) q[1];
cx q[0],q[1];
ry(3.065059010324063) q[2];
ry(2.7045130321758606) q[3];
cx q[2],q[3];
ry(0.37160431446342373) q[2];
ry(-1.2691809012862656) q[3];
cx q[2],q[3];
ry(-1.6271577565570954) q[4];
ry(-1.7485757990460264) q[5];
cx q[4],q[5];
ry(-2.9947672599048754) q[4];
ry(1.5551433517571762) q[5];
cx q[4],q[5];
ry(-0.2600847221579301) q[6];
ry(1.1711584449770114) q[7];
cx q[6],q[7];
ry(-2.216967440872666) q[6];
ry(-2.2175295528859635) q[7];
cx q[6],q[7];
ry(-0.7815387823846462) q[1];
ry(-0.6183657575559156) q[2];
cx q[1],q[2];
ry(2.1565803805481405) q[1];
ry(1.242496650264449) q[2];
cx q[1],q[2];
ry(-3.0387106626632887) q[3];
ry(-3.0595446296993143) q[4];
cx q[3],q[4];
ry(1.2085098214287937) q[3];
ry(-0.6762664599032318) q[4];
cx q[3],q[4];
ry(-1.8858798186841974) q[5];
ry(0.3793189822769545) q[6];
cx q[5],q[6];
ry(-1.7478760152340118) q[5];
ry(0.8853407946023637) q[6];
cx q[5],q[6];
ry(0.7964869739575269) q[0];
ry(-1.97325033677625) q[1];
cx q[0],q[1];
ry(-0.7980174788130818) q[0];
ry(-1.6225659159134203) q[1];
cx q[0],q[1];
ry(-1.6345735196027267) q[2];
ry(1.462263659199726) q[3];
cx q[2],q[3];
ry(-0.9902208542944558) q[2];
ry(-2.7762034056415006) q[3];
cx q[2],q[3];
ry(-2.1740493354315875) q[4];
ry(2.1850757657378814) q[5];
cx q[4],q[5];
ry(2.479284346288954) q[4];
ry(0.18970363027725146) q[5];
cx q[4],q[5];
ry(-3.1238555305909044) q[6];
ry(2.088819129147486) q[7];
cx q[6],q[7];
ry(-0.8391650820989405) q[6];
ry(2.528666165721063) q[7];
cx q[6],q[7];
ry(-0.6158470290731373) q[1];
ry(-2.4717365050032254) q[2];
cx q[1],q[2];
ry(1.2241026785785314) q[1];
ry(1.7414739236648025) q[2];
cx q[1],q[2];
ry(-2.497540764010855) q[3];
ry(-1.4904324580506367) q[4];
cx q[3],q[4];
ry(-1.3483268122251355) q[3];
ry(1.7105089920592804) q[4];
cx q[3],q[4];
ry(-1.7289505474847005) q[5];
ry(0.34110964253212117) q[6];
cx q[5],q[6];
ry(2.1411977241239226) q[5];
ry(-1.8966014001394966) q[6];
cx q[5],q[6];
ry(2.860992793409391) q[0];
ry(-2.7129958893713693) q[1];
cx q[0],q[1];
ry(-0.8315773047756129) q[0];
ry(-0.7939016889012314) q[1];
cx q[0],q[1];
ry(0.13239582408033443) q[2];
ry(-1.5964523908950365) q[3];
cx q[2],q[3];
ry(0.6567963662024994) q[2];
ry(2.382114315443468) q[3];
cx q[2],q[3];
ry(-0.9897309895309379) q[4];
ry(-1.9136401191682741) q[5];
cx q[4],q[5];
ry(0.8546889987115271) q[4];
ry(-0.015266639879003563) q[5];
cx q[4],q[5];
ry(-2.611260249383939) q[6];
ry(-2.150475460001969) q[7];
cx q[6],q[7];
ry(0.6747530446317204) q[6];
ry(-0.1785208070110369) q[7];
cx q[6],q[7];
ry(-1.1975957081556459) q[1];
ry(0.05079452996888989) q[2];
cx q[1],q[2];
ry(-1.4345428717743045) q[1];
ry(0.8053081803489395) q[2];
cx q[1],q[2];
ry(2.4518515252815485) q[3];
ry(-1.1320603873707837) q[4];
cx q[3],q[4];
ry(-0.4225416699780713) q[3];
ry(-2.381362261699351) q[4];
cx q[3],q[4];
ry(-0.33280401260864223) q[5];
ry(-1.0932576118141972) q[6];
cx q[5],q[6];
ry(2.4961412933369167) q[5];
ry(2.8646482567825586) q[6];
cx q[5],q[6];
ry(2.8333616722556556) q[0];
ry(1.0995793548280703) q[1];
cx q[0],q[1];
ry(-0.49561278577432344) q[0];
ry(-1.165489896897209) q[1];
cx q[0],q[1];
ry(0.41404227313697817) q[2];
ry(-2.6908259177163707) q[3];
cx q[2],q[3];
ry(-2.2320997891543026) q[2];
ry(2.3159138523630514) q[3];
cx q[2],q[3];
ry(0.47195925586279197) q[4];
ry(0.48290852323025657) q[5];
cx q[4],q[5];
ry(-0.1860979176746943) q[4];
ry(-1.0957849470113112) q[5];
cx q[4],q[5];
ry(-2.91587371162959) q[6];
ry(2.02328008735678) q[7];
cx q[6],q[7];
ry(1.7662137267771305) q[6];
ry(1.1082449385655504) q[7];
cx q[6],q[7];
ry(-0.3446566438519009) q[1];
ry(1.0137172841385924) q[2];
cx q[1],q[2];
ry(-1.0166481439094213) q[1];
ry(-1.3557612723485248) q[2];
cx q[1],q[2];
ry(-1.2754409337656134) q[3];
ry(1.8340453091440478) q[4];
cx q[3],q[4];
ry(1.2613344075154496) q[3];
ry(-1.5586236520115584) q[4];
cx q[3],q[4];
ry(-1.408062353866936) q[5];
ry(2.1533398523369813) q[6];
cx q[5],q[6];
ry(1.915810065816153) q[5];
ry(0.1710251228876121) q[6];
cx q[5],q[6];
ry(-1.4986742831086175) q[0];
ry(-3.0516736041702703) q[1];
cx q[0],q[1];
ry(-3.0758616554958946) q[0];
ry(1.618830010418213) q[1];
cx q[0],q[1];
ry(1.5304883187637923) q[2];
ry(-1.8570578708475918) q[3];
cx q[2],q[3];
ry(0.6620080431594266) q[2];
ry(2.6199089824834285) q[3];
cx q[2],q[3];
ry(2.7964124045930783) q[4];
ry(-0.563976059620904) q[5];
cx q[4],q[5];
ry(0.2606335799983368) q[4];
ry(0.8229906301033534) q[5];
cx q[4],q[5];
ry(2.42538840940111) q[6];
ry(-1.6673456849293684) q[7];
cx q[6],q[7];
ry(0.6410611955856221) q[6];
ry(-1.5283356675792337) q[7];
cx q[6],q[7];
ry(-2.825027315573611) q[1];
ry(-1.5846971200605502) q[2];
cx q[1],q[2];
ry(-3.0996425808384296) q[1];
ry(-0.15707130105133563) q[2];
cx q[1],q[2];
ry(3.1102931813753414) q[3];
ry(-0.07440575691670848) q[4];
cx q[3],q[4];
ry(2.6288049525447152) q[3];
ry(-2.921765270513034) q[4];
cx q[3],q[4];
ry(1.773595150199834) q[5];
ry(-1.0348626908751601) q[6];
cx q[5],q[6];
ry(-0.7200793223922002) q[5];
ry(-2.204236310573777) q[6];
cx q[5],q[6];
ry(-2.327783269845747) q[0];
ry(1.1419261290866096) q[1];
cx q[0],q[1];
ry(-1.6935138757535428) q[0];
ry(2.37032197835914) q[1];
cx q[0],q[1];
ry(2.3223561700344515) q[2];
ry(-0.4262678884698881) q[3];
cx q[2],q[3];
ry(1.8698356594982943) q[2];
ry(-2.5073703387906683) q[3];
cx q[2],q[3];
ry(-1.6450232982000217) q[4];
ry(2.807508558881082) q[5];
cx q[4],q[5];
ry(3.0097942705704184) q[4];
ry(-1.782000571128634) q[5];
cx q[4],q[5];
ry(-1.9972511356505196) q[6];
ry(-1.6186622134683992) q[7];
cx q[6],q[7];
ry(0.8346018915427591) q[6];
ry(0.9225947654191011) q[7];
cx q[6],q[7];
ry(-0.8415320597212116) q[1];
ry(-1.4231005550804152) q[2];
cx q[1],q[2];
ry(2.4088111857189136) q[1];
ry(-2.5366217794460444) q[2];
cx q[1],q[2];
ry(1.3861576722682643) q[3];
ry(-1.8794503598144097) q[4];
cx q[3],q[4];
ry(2.5457513644904415) q[3];
ry(2.6412389607387765) q[4];
cx q[3],q[4];
ry(-3.0137473087913853) q[5];
ry(-1.810776331150902) q[6];
cx q[5],q[6];
ry(-0.08120881623482412) q[5];
ry(1.4111638027538609) q[6];
cx q[5],q[6];
ry(-0.29984512118257833) q[0];
ry(-2.978627621618225) q[1];
cx q[0],q[1];
ry(2.8646798275531045) q[0];
ry(1.1397741418288643) q[1];
cx q[0],q[1];
ry(1.8335560974167544) q[2];
ry(1.3353647590310405) q[3];
cx q[2],q[3];
ry(-2.324312753022722) q[2];
ry(-2.0919853359587117) q[3];
cx q[2],q[3];
ry(-2.3827832684298342) q[4];
ry(0.8823758902220247) q[5];
cx q[4],q[5];
ry(-3.0515129985648173) q[4];
ry(2.044105612581774) q[5];
cx q[4],q[5];
ry(0.26168998853221653) q[6];
ry(-1.0462058685647917) q[7];
cx q[6],q[7];
ry(-2.5578226883368282) q[6];
ry(0.8334332838857267) q[7];
cx q[6],q[7];
ry(3.0967827676089823) q[1];
ry(-2.804597528374744) q[2];
cx q[1],q[2];
ry(-0.12724043904145613) q[1];
ry(2.5944993886171024) q[2];
cx q[1],q[2];
ry(-0.580988247275064) q[3];
ry(2.6603063231300563) q[4];
cx q[3],q[4];
ry(-2.381477752687787) q[3];
ry(-1.90800262017278) q[4];
cx q[3],q[4];
ry(-1.5479212619396163) q[5];
ry(-1.8593423708028467) q[6];
cx q[5],q[6];
ry(-2.6133382446499174) q[5];
ry(1.678601276834223) q[6];
cx q[5],q[6];
ry(1.3515529228348804) q[0];
ry(2.579920704872338) q[1];
cx q[0],q[1];
ry(-3.0301785582774006) q[0];
ry(-1.3317153901829997) q[1];
cx q[0],q[1];
ry(0.45592997266256496) q[2];
ry(1.6658956314397646) q[3];
cx q[2],q[3];
ry(-0.32476045491776734) q[2];
ry(-1.374555326504474) q[3];
cx q[2],q[3];
ry(-1.9428856150156282) q[4];
ry(-2.9107986657898195) q[5];
cx q[4],q[5];
ry(-3.1011385428638287) q[4];
ry(-2.713146492227752) q[5];
cx q[4],q[5];
ry(1.0724103997197134) q[6];
ry(-0.2906063340368909) q[7];
cx q[6],q[7];
ry(-1.9914186972472445) q[6];
ry(-2.5687959379144423) q[7];
cx q[6],q[7];
ry(-1.4595936295478946) q[1];
ry(-2.3448004841746055) q[2];
cx q[1],q[2];
ry(-2.439655840985774) q[1];
ry(3.124123143593791) q[2];
cx q[1],q[2];
ry(2.855251385848158) q[3];
ry(-1.8778451478361458) q[4];
cx q[3],q[4];
ry(-2.3036911819833) q[3];
ry(-2.3030270659456447) q[4];
cx q[3],q[4];
ry(2.774606466217569) q[5];
ry(-1.6085547721168572) q[6];
cx q[5],q[6];
ry(-2.158887836720096) q[5];
ry(2.0862251193574064) q[6];
cx q[5],q[6];
ry(-2.6020766628046452) q[0];
ry(1.2360907625265698) q[1];
cx q[0],q[1];
ry(1.2224746199053833) q[0];
ry(-2.2969511217854715) q[1];
cx q[0],q[1];
ry(-2.3311573864065354) q[2];
ry(2.0810789137688355) q[3];
cx q[2],q[3];
ry(2.778769411464537) q[2];
ry(2.784816233505246) q[3];
cx q[2],q[3];
ry(-1.9572710719397124) q[4];
ry(-2.4176985805996125) q[5];
cx q[4],q[5];
ry(2.7799925688399165) q[4];
ry(2.32751379679355) q[5];
cx q[4],q[5];
ry(-2.4810898129553913) q[6];
ry(-2.295553632184992) q[7];
cx q[6],q[7];
ry(-0.6589087490159957) q[6];
ry(-1.6982942410413686) q[7];
cx q[6],q[7];
ry(1.689991661236757) q[1];
ry(1.1677629176779716) q[2];
cx q[1],q[2];
ry(-1.287646806371745) q[1];
ry(2.3662153970481676) q[2];
cx q[1],q[2];
ry(0.7415884298295037) q[3];
ry(-1.613765815002215) q[4];
cx q[3],q[4];
ry(-2.338108124617154) q[3];
ry(-1.5609625164545753) q[4];
cx q[3],q[4];
ry(-0.6921849787301637) q[5];
ry(2.0377330251121686) q[6];
cx q[5],q[6];
ry(0.032080704701117346) q[5];
ry(2.2398758963327667) q[6];
cx q[5],q[6];
ry(-0.7697309981942188) q[0];
ry(-1.6862243146055498) q[1];
cx q[0],q[1];
ry(0.7417907149455827) q[0];
ry(-0.6770802961004659) q[1];
cx q[0],q[1];
ry(-2.3430435371540232) q[2];
ry(3.0013664469039396) q[3];
cx q[2],q[3];
ry(0.7558937154356995) q[2];
ry(-0.4243141601320934) q[3];
cx q[2],q[3];
ry(2.5030311461988877) q[4];
ry(1.1115162625636268) q[5];
cx q[4],q[5];
ry(2.708962015030165) q[4];
ry(-0.9563862329067209) q[5];
cx q[4],q[5];
ry(-0.01679471531126442) q[6];
ry(3.0501839002664544) q[7];
cx q[6],q[7];
ry(2.4464189034892128) q[6];
ry(-3.047489673090141) q[7];
cx q[6],q[7];
ry(-0.7588003269650532) q[1];
ry(0.6670914038214768) q[2];
cx q[1],q[2];
ry(0.6764503129027496) q[1];
ry(1.8599792328934759) q[2];
cx q[1],q[2];
ry(-2.7982226795033633) q[3];
ry(-2.5592551796342815) q[4];
cx q[3],q[4];
ry(-0.252503098489675) q[3];
ry(-0.033693491079058724) q[4];
cx q[3],q[4];
ry(1.7973261605323807) q[5];
ry(2.521260720804003) q[6];
cx q[5],q[6];
ry(0.4180332878118724) q[5];
ry(-2.1224031466451643) q[6];
cx q[5],q[6];
ry(-1.1679271316653175) q[0];
ry(0.5085346640087902) q[1];
cx q[0],q[1];
ry(0.19377502863030927) q[0];
ry(-2.5019839200070098) q[1];
cx q[0],q[1];
ry(-0.5075655460552531) q[2];
ry(0.04730899574279369) q[3];
cx q[2],q[3];
ry(0.14589148124999973) q[2];
ry(-0.8261184255550129) q[3];
cx q[2],q[3];
ry(-2.1837561240385908) q[4];
ry(-1.908858161103119) q[5];
cx q[4],q[5];
ry(2.017551808279955) q[4];
ry(1.4345164821907366) q[5];
cx q[4],q[5];
ry(-1.8898541651411431) q[6];
ry(0.11072632668433412) q[7];
cx q[6],q[7];
ry(-2.9414803036457577) q[6];
ry(-0.5079868195418991) q[7];
cx q[6],q[7];
ry(-1.3977736869291997) q[1];
ry(-0.8273406474970075) q[2];
cx q[1],q[2];
ry(-2.1491270316066697) q[1];
ry(-1.6520071818336213) q[2];
cx q[1],q[2];
ry(-1.9503536447643997) q[3];
ry(2.8573308844103633) q[4];
cx q[3],q[4];
ry(-1.222991335641999) q[3];
ry(-0.36100234459156066) q[4];
cx q[3],q[4];
ry(-1.4237239945716782) q[5];
ry(-1.9391149102385423) q[6];
cx q[5],q[6];
ry(2.2189263807021495) q[5];
ry(-0.8484800780219414) q[6];
cx q[5],q[6];
ry(-2.3512891160748746) q[0];
ry(2.4774517025887843) q[1];
cx q[0],q[1];
ry(-0.6649441836188574) q[0];
ry(2.5758131924787304) q[1];
cx q[0],q[1];
ry(1.1177516755195152) q[2];
ry(1.2710840514255184) q[3];
cx q[2],q[3];
ry(0.831744512741869) q[2];
ry(-1.8116465845613445) q[3];
cx q[2],q[3];
ry(-0.7011779072711265) q[4];
ry(-0.7478271893405777) q[5];
cx q[4],q[5];
ry(1.4939644792121738) q[4];
ry(-2.198591582918233) q[5];
cx q[4],q[5];
ry(1.2673060934190854) q[6];
ry(2.0056852230984035) q[7];
cx q[6],q[7];
ry(-0.22356510416992406) q[6];
ry(-0.4258526574442216) q[7];
cx q[6],q[7];
ry(1.0853746861077775) q[1];
ry(-2.0497084554801894) q[2];
cx q[1],q[2];
ry(-2.3420610449206545) q[1];
ry(0.04925046167287215) q[2];
cx q[1],q[2];
ry(-0.9746744275901753) q[3];
ry(-2.384789386929399) q[4];
cx q[3],q[4];
ry(-1.9778858805423258) q[3];
ry(-2.7576411692576084) q[4];
cx q[3],q[4];
ry(1.8793978358443595) q[5];
ry(-0.6386437783286432) q[6];
cx q[5],q[6];
ry(-0.6068306669260668) q[5];
ry(-1.0205108691034925) q[6];
cx q[5],q[6];
ry(2.197416040092513) q[0];
ry(1.5711566605005718) q[1];
cx q[0],q[1];
ry(0.42878757678695817) q[0];
ry(3.022732876379834) q[1];
cx q[0],q[1];
ry(3.066469741311561) q[2];
ry(0.45732171832364654) q[3];
cx q[2],q[3];
ry(0.7673038969722477) q[2];
ry(1.6670147343043418) q[3];
cx q[2],q[3];
ry(1.1423472215004482) q[4];
ry(-0.8051790364949998) q[5];
cx q[4],q[5];
ry(-0.6889947939733334) q[4];
ry(-1.681230926751411) q[5];
cx q[4],q[5];
ry(1.4152067409389544) q[6];
ry(-3.0440322030112594) q[7];
cx q[6],q[7];
ry(-1.697883818157403) q[6];
ry(0.03820913394003966) q[7];
cx q[6],q[7];
ry(-1.3553644554798538) q[1];
ry(-0.6355080298299529) q[2];
cx q[1],q[2];
ry(0.024411602955656038) q[1];
ry(1.7885560680040422) q[2];
cx q[1],q[2];
ry(-2.139568766064232) q[3];
ry(1.77721135597909) q[4];
cx q[3],q[4];
ry(0.1710196514099797) q[3];
ry(1.35898384385505) q[4];
cx q[3],q[4];
ry(-1.0906285997000609) q[5];
ry(-2.9066130546436315) q[6];
cx q[5],q[6];
ry(-2.4712245832280932) q[5];
ry(-1.3802646953607356) q[6];
cx q[5],q[6];
ry(3.057314189407183) q[0];
ry(-1.0117355465660864) q[1];
cx q[0],q[1];
ry(-0.773717357381682) q[0];
ry(1.3788211433981883) q[1];
cx q[0],q[1];
ry(0.485568312278589) q[2];
ry(-1.7007761077981467) q[3];
cx q[2],q[3];
ry(-1.3274884736313854) q[2];
ry(-2.8478472163503055) q[3];
cx q[2],q[3];
ry(-2.6202050849772) q[4];
ry(-1.0393302030224811) q[5];
cx q[4],q[5];
ry(-1.6308582765311528) q[4];
ry(-2.864151961298391) q[5];
cx q[4],q[5];
ry(0.469762203435577) q[6];
ry(-2.125448575757127) q[7];
cx q[6],q[7];
ry(2.227245559810927) q[6];
ry(0.46454666133350225) q[7];
cx q[6],q[7];
ry(1.1038960463475316) q[1];
ry(-2.5054013841229907) q[2];
cx q[1],q[2];
ry(-1.0447667962352662) q[1];
ry(-2.0556590560198384) q[2];
cx q[1],q[2];
ry(0.20200815151711893) q[3];
ry(-1.9163923405740588) q[4];
cx q[3],q[4];
ry(1.4507171196424629) q[3];
ry(2.0301346818898067) q[4];
cx q[3],q[4];
ry(1.011019612179636) q[5];
ry(-2.3819732674957557) q[6];
cx q[5],q[6];
ry(1.3080207094913467) q[5];
ry(-0.4808472308753613) q[6];
cx q[5],q[6];
ry(-0.31871044197749077) q[0];
ry(-2.261597644428655) q[1];
cx q[0],q[1];
ry(0.445258007393071) q[0];
ry(0.8848226037393551) q[1];
cx q[0],q[1];
ry(0.8680144415912875) q[2];
ry(-2.9026382293722364) q[3];
cx q[2],q[3];
ry(0.03990333414830438) q[2];
ry(-2.3645372801530784) q[3];
cx q[2],q[3];
ry(1.1952638410930003) q[4];
ry(-0.32220795112328204) q[5];
cx q[4],q[5];
ry(-2.704992210125602) q[4];
ry(1.890593272619653) q[5];
cx q[4],q[5];
ry(0.7648792387878979) q[6];
ry(-0.5697901315592259) q[7];
cx q[6],q[7];
ry(-1.6132198136135052) q[6];
ry(0.12286912743017729) q[7];
cx q[6],q[7];
ry(2.0124231828879298) q[1];
ry(-2.248708763860157) q[2];
cx q[1],q[2];
ry(2.7995550103340445) q[1];
ry(-2.5103406301157114) q[2];
cx q[1],q[2];
ry(2.0249554056028476) q[3];
ry(0.7606907385079023) q[4];
cx q[3],q[4];
ry(2.7592156785951696) q[3];
ry(-1.7703558241621082) q[4];
cx q[3],q[4];
ry(0.7809632297290312) q[5];
ry(1.375889769462053) q[6];
cx q[5],q[6];
ry(-1.3024553742482619) q[5];
ry(-1.780057836499791) q[6];
cx q[5],q[6];
ry(1.0465753716529438) q[0];
ry(-1.1885444852385358) q[1];
cx q[0],q[1];
ry(1.8298494884862466) q[0];
ry(-0.4678483113277583) q[1];
cx q[0],q[1];
ry(2.087418509381549) q[2];
ry(0.92604983013861) q[3];
cx q[2],q[3];
ry(0.2341030300987378) q[2];
ry(-3.026532476352438) q[3];
cx q[2],q[3];
ry(0.685122923508545) q[4];
ry(-1.0792032223427406) q[5];
cx q[4],q[5];
ry(0.20332515591143424) q[4];
ry(-1.3277253317471815) q[5];
cx q[4],q[5];
ry(1.7384047871831614) q[6];
ry(0.45920977917036715) q[7];
cx q[6],q[7];
ry(-2.7678842411973386) q[6];
ry(-0.5218020885780693) q[7];
cx q[6],q[7];
ry(-2.24057033315645) q[1];
ry(-1.2253863301804229) q[2];
cx q[1],q[2];
ry(-2.340357938619167) q[1];
ry(-0.652188310568186) q[2];
cx q[1],q[2];
ry(0.7828528877082261) q[3];
ry(-0.013227682349825189) q[4];
cx q[3],q[4];
ry(-1.8383790177153136) q[3];
ry(-1.0807055840089044) q[4];
cx q[3],q[4];
ry(-2.8924614389625223) q[5];
ry(2.2188249676313787) q[6];
cx q[5],q[6];
ry(-0.7851307061317803) q[5];
ry(-0.7727315973040074) q[6];
cx q[5],q[6];
ry(-2.6898208207439813) q[0];
ry(2.511726207063678) q[1];
ry(0.4794686585806387) q[2];
ry(1.3263901050118023) q[3];
ry(0.8012064647705703) q[4];
ry(3.069133538968455) q[5];
ry(2.2088962739311353) q[6];
ry(-2.799826258275813) q[7];