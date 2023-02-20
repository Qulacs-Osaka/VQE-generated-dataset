OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.146632406809693) q[0];
rz(1.808050780663197) q[0];
ry(2.7786978083415614) q[1];
rz(1.1832062442542655) q[1];
ry(-0.6131458904534206) q[2];
rz(1.9017756589190595) q[2];
ry(0.8387622240387961) q[3];
rz(0.31623731862038096) q[3];
ry(-2.8683965252092936) q[4];
rz(-1.380607118617604) q[4];
ry(3.0106124140295956) q[5];
rz(1.2000787601141862) q[5];
ry(1.251087574593973) q[6];
rz(0.8155719385554747) q[6];
ry(1.076024383711962) q[7];
rz(0.4671321813237163) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5206489489682993) q[0];
rz(1.3287723896946875) q[0];
ry(-1.9923744818186655) q[1];
rz(-0.35545626523230567) q[1];
ry(-0.3247728640739931) q[2];
rz(1.7262268494916087) q[2];
ry(-2.3640766034761636) q[3];
rz(-2.0844696478515647) q[3];
ry(-2.477610716070056) q[4];
rz(0.017393782696577605) q[4];
ry(2.538916465096139) q[5];
rz(-2.212924391364636) q[5];
ry(-1.6623718198761213) q[6];
rz(-1.4620752948093871) q[6];
ry(-2.6405138640373145) q[7];
rz(2.911052216630994) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.357369509685248) q[0];
rz(2.912387119283169) q[0];
ry(-2.6662889768062326) q[1];
rz(1.1829596449700448) q[1];
ry(-2.927148796536055) q[2];
rz(-0.9581423479439626) q[2];
ry(-2.0531536718958536) q[3];
rz(0.9656229462591178) q[3];
ry(0.8058013665295668) q[4];
rz(2.6462452566223607) q[4];
ry(1.8367092692185105) q[5];
rz(0.4559053793218402) q[5];
ry(2.6984172853160806) q[6];
rz(-0.7871661682287191) q[6];
ry(2.539320487144103) q[7];
rz(-2.3874111983520017) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.3334459368852585) q[0];
rz(-2.300419476677734) q[0];
ry(-2.314441979487267) q[1];
rz(-0.4329934106200355) q[1];
ry(0.2666198775921087) q[2];
rz(-0.7865165534560911) q[2];
ry(-3.0547853142743366) q[3];
rz(-2.2767627597947593) q[3];
ry(-0.747980427872911) q[4];
rz(-1.9202054584208579) q[4];
ry(-0.8561827951662035) q[5];
rz(-0.018157904480784026) q[5];
ry(-2.3355669378803197) q[6];
rz(1.2311135968920868) q[6];
ry(-2.148483337055072) q[7];
rz(-2.8381618137111024) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.525265187308603) q[0];
rz(0.10590580716335028) q[0];
ry(-1.7176255414298789) q[1];
rz(-2.4575434036513206) q[1];
ry(0.5663732648850054) q[2];
rz(-1.6495802904530172) q[2];
ry(2.459069640650748) q[3];
rz(-3.030028199468092) q[3];
ry(-0.7966274105350539) q[4];
rz(0.43258033514881333) q[4];
ry(-0.9881198795637696) q[5];
rz(2.910935488949771) q[5];
ry(-1.4408278996097934) q[6];
rz(1.56736360031546) q[6];
ry(1.7136245589787302) q[7];
rz(-1.9713381220518953) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.6879917801163813) q[0];
rz(-0.300049872083366) q[0];
ry(2.6342040906850235) q[1];
rz(0.5579109417321905) q[1];
ry(1.8409398996296016) q[2];
rz(-2.2809355188862828) q[2];
ry(-0.3303920717648593) q[3];
rz(2.7913066605709558) q[3];
ry(3.0386245954655338) q[4];
rz(0.9838780037107268) q[4];
ry(2.2982614782585835) q[5];
rz(0.7143842101352771) q[5];
ry(-0.11384630489245116) q[6];
rz(-0.7764356425397577) q[6];
ry(2.217769888213849) q[7];
rz(-0.8791214722478434) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.0327588823429992) q[0];
rz(-2.31949156263861) q[0];
ry(-0.5354188030135294) q[1];
rz(-2.0903449151486324) q[1];
ry(-2.9535739314611047) q[2];
rz(2.5227991992031695) q[2];
ry(1.5739190000673124) q[3];
rz(-0.818999585552717) q[3];
ry(1.171977614854991) q[4];
rz(-0.9089470028870315) q[4];
ry(2.088872766306019) q[5];
rz(-1.8021111324889167) q[5];
ry(2.104742101837531) q[6];
rz(-1.8732291818022322) q[6];
ry(1.6377690039865085) q[7];
rz(1.76935964590173) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.720936895164941) q[0];
rz(1.351435709009003) q[0];
ry(-3.033290365129011) q[1];
rz(-0.7159621992407253) q[1];
ry(1.154042672361296) q[2];
rz(-2.109072684589406) q[2];
ry(1.4709573000447989) q[3];
rz(-1.8062530187608283) q[3];
ry(2.4985858045417912) q[4];
rz(-2.4272354216130636) q[4];
ry(0.9274918739915917) q[5];
rz(-3.0583701437326383) q[5];
ry(-2.544809041993741) q[6];
rz(2.638452319688726) q[6];
ry(0.567401158436577) q[7];
rz(1.2561577378375972) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.2393577847835218) q[0];
rz(1.3667523338733192) q[0];
ry(2.85451800633232) q[1];
rz(-1.1080610656439522) q[1];
ry(2.0728633514987442) q[2];
rz(-2.8829645071586603) q[2];
ry(-0.6722733871185533) q[3];
rz(-1.5006727342671002) q[3];
ry(-1.7183065828719881) q[4];
rz(-2.973965317624274) q[4];
ry(-0.24634051856928096) q[5];
rz(0.17421865395248148) q[5];
ry(2.0252839406973164) q[6];
rz(-2.2553659170882394) q[6];
ry(-2.29362777815446) q[7];
rz(2.3777263875564625) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4580741767013485) q[0];
rz(-0.12379263600674056) q[0];
ry(-2.8684685399661203) q[1];
rz(1.4814579331161333) q[1];
ry(-1.9691641714898678) q[2];
rz(-1.4835545244349584) q[2];
ry(0.9409288117858391) q[3];
rz(-1.507093270994316) q[3];
ry(2.446566133006532) q[4];
rz(-0.6753190694288334) q[4];
ry(-1.0719402229064239) q[5];
rz(3.0090396292131496) q[5];
ry(2.7740142385451083) q[6];
rz(1.9123725861473375) q[6];
ry(-0.2768497430806063) q[7];
rz(-1.046020589062726) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.972191101822863) q[0];
rz(2.6281414032192774) q[0];
ry(-2.408182586979784) q[1];
rz(-0.6347131075939294) q[1];
ry(2.714526105436704) q[2];
rz(0.7715865788671223) q[2];
ry(-0.19690226372423325) q[3];
rz(0.01589953336941363) q[3];
ry(2.3816866493189237) q[4];
rz(-0.384758441375772) q[4];
ry(0.4655278472828773) q[5];
rz(2.3326440628255756) q[5];
ry(2.6966996814287025) q[6];
rz(0.9217371444635445) q[6];
ry(-1.6660551073423218) q[7];
rz(-1.5069879715402166) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.5991553105589783) q[0];
rz(-0.7646916436348474) q[0];
ry(-2.9871070527101775) q[1];
rz(-0.36426786420251156) q[1];
ry(1.334418055810815) q[2];
rz(1.8435701122524062) q[2];
ry(2.138938192326912) q[3];
rz(-0.9448419108206698) q[3];
ry(1.8375871567192017) q[4];
rz(1.1580689697109205) q[4];
ry(-2.402949746390146) q[5];
rz(-1.5924051225612474) q[5];
ry(-1.4544667349935256) q[6];
rz(-0.5936931679033494) q[6];
ry(1.23767998477489) q[7];
rz(2.3097392101444005) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.3733008316860587) q[0];
rz(0.45419789288851936) q[0];
ry(0.34150701412866796) q[1];
rz(-0.4622867734238983) q[1];
ry(-2.385834379401917) q[2];
rz(-2.3424023866366563) q[2];
ry(0.19002432122108726) q[3];
rz(1.7082800921294536) q[3];
ry(1.6674793926057716) q[4];
rz(1.958652990232742) q[4];
ry(-1.2974150520670955) q[5];
rz(2.4770617376550255) q[5];
ry(1.0606945253068951) q[6];
rz(3.131773445328613) q[6];
ry(2.5690818765659253) q[7];
rz(2.2709129055687125) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6090258628690473) q[0];
rz(1.0516632682378848) q[0];
ry(0.3958774316383472) q[1];
rz(-2.282435304652825) q[1];
ry(-1.203122299532812) q[2];
rz(-1.364262170364177) q[2];
ry(1.9122724146970596) q[3];
rz(-0.8889080872288263) q[3];
ry(-0.5590019102383363) q[4];
rz(2.7838394150405463) q[4];
ry(1.8837289957062493) q[5];
rz(0.22771445698292894) q[5];
ry(-2.327858385892803) q[6];
rz(-0.975913585891636) q[6];
ry(-0.6240859035540707) q[7];
rz(-0.17672756939216502) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.8557002334591474) q[0];
rz(-2.7577746913753542) q[0];
ry(1.0286898417473145) q[1];
rz(0.7237331340068278) q[1];
ry(-0.1926649275603447) q[2];
rz(0.019910974540278122) q[2];
ry(0.6635172125853761) q[3];
rz(-2.446142876310788) q[3];
ry(-0.5232399229408106) q[4];
rz(0.7114140979157) q[4];
ry(2.7956833223961626) q[5];
rz(-1.4373095576883808) q[5];
ry(-1.745866958978092) q[6];
rz(-0.3615115616926792) q[6];
ry(-0.40392563755691047) q[7];
rz(1.650241993514253) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.27625092512106336) q[0];
rz(2.5291634482920737) q[0];
ry(0.8512881977540463) q[1];
rz(-1.7866035841538106) q[1];
ry(-0.4886953973552682) q[2];
rz(2.944482497215476) q[2];
ry(-2.2888358667936384) q[3];
rz(-2.0192109393491524) q[3];
ry(3.0073107642119044) q[4];
rz(-1.8692811923370076) q[4];
ry(1.1765096353976292) q[5];
rz(-1.1133731128519075) q[5];
ry(0.3949689490593027) q[6];
rz(2.9211798354791316) q[6];
ry(2.6255317745116047) q[7];
rz(1.9353837970160799) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.8933331259730464) q[0];
rz(-0.6427515239979752) q[0];
ry(-2.789835543204257) q[1];
rz(0.29355907811483684) q[1];
ry(-0.7974440783753356) q[2];
rz(2.2438663794642784) q[2];
ry(-2.5018396814301016) q[3];
rz(0.9383717785111517) q[3];
ry(-0.7635402174194075) q[4];
rz(1.7765126645871314) q[4];
ry(0.8812475612237836) q[5];
rz(-2.6176490915237243) q[5];
ry(-1.2813305143364706) q[6];
rz(0.05214407532778167) q[6];
ry(2.5958799832083828) q[7];
rz(0.534541013007768) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.19133832974095) q[0];
rz(-1.674659028277918) q[0];
ry(0.9956281100831045) q[1];
rz(-1.0555091354030848) q[1];
ry(-1.8482268124380277) q[2];
rz(1.9468695430866119) q[2];
ry(1.9022254034785722) q[3];
rz(-1.5473988053995011) q[3];
ry(0.7200238801900657) q[4];
rz(0.34972775418488455) q[4];
ry(2.2170250311222017) q[5];
rz(-2.6209046740326856) q[5];
ry(-1.716158778104658) q[6];
rz(-3.0815078126346185) q[6];
ry(0.6993148426605907) q[7];
rz(-2.4572653197852627) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5135609812200608) q[0];
rz(2.1277901569623987) q[0];
ry(2.0897003891322283) q[1];
rz(0.3111244048456179) q[1];
ry(-3.1286522599532627) q[2];
rz(-0.804230770737364) q[2];
ry(-0.46598792410382817) q[3];
rz(0.8158199126657771) q[3];
ry(0.682420854353306) q[4];
rz(-2.622687375226679) q[4];
ry(-1.8234202138835067) q[5];
rz(2.6882807358447005) q[5];
ry(-2.514959211102708) q[6];
rz(1.9643612458420083) q[6];
ry(-2.875043256043941) q[7];
rz(-2.9888752029985044) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.3699875393782168) q[0];
rz(3.009891734700395) q[0];
ry(-2.4014911276524216) q[1];
rz(-0.5077765906412707) q[1];
ry(0.9041253039608621) q[2];
rz(2.2024812127082067) q[2];
ry(2.819557790071785) q[3];
rz(0.27674045603974845) q[3];
ry(-1.127988502097095) q[4];
rz(2.9220329104309477) q[4];
ry(1.7126243960201215) q[5];
rz(-3.0552371543268353) q[5];
ry(2.962200519851835) q[6];
rz(0.8614793853756596) q[6];
ry(0.4593579585485787) q[7];
rz(2.9903659304663157) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.2541230953488696) q[0];
rz(-1.7771129757060034) q[0];
ry(0.13987880404728834) q[1];
rz(-0.5776453375660981) q[1];
ry(2.837952981417789) q[2];
rz(1.0065430016794572) q[2];
ry(-2.865748681890203) q[3];
rz(-2.4461032994715017) q[3];
ry(2.7274538265238415) q[4];
rz(0.8077122441849658) q[4];
ry(2.647599111738789) q[5];
rz(2.593284276324751) q[5];
ry(2.9338917496709493) q[6];
rz(-1.0989918825339595) q[6];
ry(-2.1469255302115124) q[7];
rz(-1.5369529600119833) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.9513892367124386) q[0];
rz(-2.204057956754859) q[0];
ry(-1.7705749989591155) q[1];
rz(1.380121714172626) q[1];
ry(1.8552295293943808) q[2];
rz(-2.3798857745673767) q[2];
ry(1.2991679203122553) q[3];
rz(0.7651426468103626) q[3];
ry(1.0045049266015376) q[4];
rz(2.942624442122877) q[4];
ry(1.7109305794393235) q[5];
rz(-0.3862502029718966) q[5];
ry(-0.6896908700667346) q[6];
rz(0.363440480979595) q[6];
ry(-0.18167608110817504) q[7];
rz(-3.054018479859911) q[7];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.880515222491153) q[0];
rz(0.07164986645426315) q[0];
ry(-1.5152111669065325) q[1];
rz(1.284802121366377) q[1];
ry(-2.4706742971982623) q[2];
rz(-2.610047668431882) q[2];
ry(2.221561935161174) q[3];
rz(0.18417687202611344) q[3];
ry(-0.34109228836487127) q[4];
rz(1.6938011621093667) q[4];
ry(2.177095036958399) q[5];
rz(-2.072327859580506) q[5];
ry(-1.4543408663979465) q[6];
rz(0.1600375795883453) q[6];
ry(2.0230708876251633) q[7];
rz(-2.6967585088259898) q[7];