OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.6773840660628783) q[0];
ry(-1.3299630897866752) q[1];
cx q[0],q[1];
ry(-1.9879385247369072) q[0];
ry(2.513300776119838) q[1];
cx q[0],q[1];
ry(-1.927427070525356) q[0];
ry(2.8674518994696405) q[2];
cx q[0],q[2];
ry(0.06572388395037267) q[0];
ry(0.7226161424454318) q[2];
cx q[0],q[2];
ry(0.9330361202295769) q[0];
ry(-2.292364422137111) q[3];
cx q[0],q[3];
ry(-0.9069379211497725) q[0];
ry(1.7036082908913635) q[3];
cx q[0],q[3];
ry(0.8516487056885491) q[1];
ry(-0.6157536890370867) q[2];
cx q[1],q[2];
ry(3.1019766071135537) q[1];
ry(-0.22107625867271355) q[2];
cx q[1],q[2];
ry(-1.5651227669470205) q[1];
ry(-1.2609468274824849) q[3];
cx q[1],q[3];
ry(-0.1724510373517334) q[1];
ry(1.8249936555562831) q[3];
cx q[1],q[3];
ry(2.575092212927717) q[2];
ry(-2.6743551935586654) q[3];
cx q[2],q[3];
ry(-0.29779582429589796) q[2];
ry(0.7946397486967225) q[3];
cx q[2],q[3];
ry(1.8486024064210733) q[0];
ry(1.9833250502135769) q[1];
cx q[0],q[1];
ry(1.1498597932239143) q[0];
ry(2.409691064097474) q[1];
cx q[0],q[1];
ry(2.8382912257059716) q[0];
ry(0.6942604190057802) q[2];
cx q[0],q[2];
ry(-2.3362248558629855) q[0];
ry(-0.7344882965865971) q[2];
cx q[0],q[2];
ry(2.13648977765022) q[0];
ry(-1.1528187362867517) q[3];
cx q[0],q[3];
ry(2.6443433161711547) q[0];
ry(-0.8866781527287388) q[3];
cx q[0],q[3];
ry(1.7085409401354692) q[1];
ry(-1.9299218206915194) q[2];
cx q[1],q[2];
ry(2.3622907513950806) q[1];
ry(-2.186003575246115) q[2];
cx q[1],q[2];
ry(2.3227292834633237) q[1];
ry(-2.583801646739943) q[3];
cx q[1],q[3];
ry(-0.8256410411622026) q[1];
ry(0.08055269295721516) q[3];
cx q[1],q[3];
ry(-3.074267003841716) q[2];
ry(0.8749774972739669) q[3];
cx q[2],q[3];
ry(1.469246906950695) q[2];
ry(0.4969016566416355) q[3];
cx q[2],q[3];
ry(-2.7491871259328815) q[0];
ry(-0.7309059192814686) q[1];
cx q[0],q[1];
ry(-3.082343389807031) q[0];
ry(-1.9184820044297517) q[1];
cx q[0],q[1];
ry(1.8626463779819211) q[0];
ry(2.7709776768279344) q[2];
cx q[0],q[2];
ry(-1.2941199963415686) q[0];
ry(-1.699972270498596) q[2];
cx q[0],q[2];
ry(-1.051861322701206) q[0];
ry(-1.5432869803218363) q[3];
cx q[0],q[3];
ry(2.0850976133121275) q[0];
ry(0.7633694357816747) q[3];
cx q[0],q[3];
ry(-1.835491359856325) q[1];
ry(0.81110760933728) q[2];
cx q[1],q[2];
ry(3.0362138675065613) q[1];
ry(-1.9082432856962421) q[2];
cx q[1],q[2];
ry(2.6954415834815317) q[1];
ry(-1.8189540509474906) q[3];
cx q[1],q[3];
ry(-1.2373323620608847) q[1];
ry(-2.9756944273694517) q[3];
cx q[1],q[3];
ry(1.9080437057731556) q[2];
ry(0.3260455729635563) q[3];
cx q[2],q[3];
ry(0.838065699051363) q[2];
ry(0.764149486955456) q[3];
cx q[2],q[3];
ry(-0.3237098829114667) q[0];
ry(-0.16200084553354888) q[1];
cx q[0],q[1];
ry(-0.2299848889497086) q[0];
ry(0.4870851021440412) q[1];
cx q[0],q[1];
ry(-0.8678376501064815) q[0];
ry(-1.6401226127507602) q[2];
cx q[0],q[2];
ry(1.9178371060160782) q[0];
ry(-1.7258648782662633) q[2];
cx q[0],q[2];
ry(0.597654789686044) q[0];
ry(0.6159628295269722) q[3];
cx q[0],q[3];
ry(3.0944094641175717) q[0];
ry(1.2798893021757243) q[3];
cx q[0],q[3];
ry(-2.132329019569075) q[1];
ry(-0.5131563110447661) q[2];
cx q[1],q[2];
ry(0.6749133332462971) q[1];
ry(-2.1911048361620074) q[2];
cx q[1],q[2];
ry(0.40694000422721555) q[1];
ry(-1.3205608140438507) q[3];
cx q[1],q[3];
ry(2.056750879114338) q[1];
ry(2.0363740864258726) q[3];
cx q[1],q[3];
ry(-0.055573103168240934) q[2];
ry(0.3935061603814072) q[3];
cx q[2],q[3];
ry(2.0731627216412836) q[2];
ry(0.041023131317896123) q[3];
cx q[2],q[3];
ry(0.858811498981007) q[0];
ry(-0.0045644502975399) q[1];
cx q[0],q[1];
ry(-2.7870282002366977) q[0];
ry(-2.607871034590325) q[1];
cx q[0],q[1];
ry(1.0276964837897804) q[0];
ry(-1.8428476382595465) q[2];
cx q[0],q[2];
ry(-0.8675183941734357) q[0];
ry(-3.016056953090513) q[2];
cx q[0],q[2];
ry(0.781435474774316) q[0];
ry(-1.2542250565192559) q[3];
cx q[0],q[3];
ry(-2.862222546007173) q[0];
ry(1.2214019660154152) q[3];
cx q[0],q[3];
ry(0.44037631129730226) q[1];
ry(-0.07729724543046146) q[2];
cx q[1],q[2];
ry(-1.7245293987711867) q[1];
ry(0.62381686999798) q[2];
cx q[1],q[2];
ry(0.4700602604279317) q[1];
ry(0.1795920928878491) q[3];
cx q[1],q[3];
ry(2.5661508052909925) q[1];
ry(-0.6907462358248311) q[3];
cx q[1],q[3];
ry(0.7084795549217454) q[2];
ry(-0.6550473770610798) q[3];
cx q[2],q[3];
ry(-1.1955805773250496) q[2];
ry(1.732686368788526) q[3];
cx q[2],q[3];
ry(2.7474909628008977) q[0];
ry(-2.294528844005365) q[1];
cx q[0],q[1];
ry(2.384318609699005) q[0];
ry(1.0507348709929334) q[1];
cx q[0],q[1];
ry(-1.0742515820568523) q[0];
ry(-2.8894938124396097) q[2];
cx q[0],q[2];
ry(2.7326566741742986) q[0];
ry(-1.696800037831883) q[2];
cx q[0],q[2];
ry(2.565760172928236) q[0];
ry(0.4681231383454358) q[3];
cx q[0],q[3];
ry(1.8793776227508197) q[0];
ry(-0.04417083223981445) q[3];
cx q[0],q[3];
ry(-3.065200668180953) q[1];
ry(0.9532210632289441) q[2];
cx q[1],q[2];
ry(-0.9859516823353218) q[1];
ry(3.1230052475801973) q[2];
cx q[1],q[2];
ry(2.184142622322304) q[1];
ry(2.6265172457504606) q[3];
cx q[1],q[3];
ry(0.9073487627269697) q[1];
ry(0.8159956247942863) q[3];
cx q[1],q[3];
ry(2.5818923243626597) q[2];
ry(3.003628898556807) q[3];
cx q[2],q[3];
ry(-2.696007080739961) q[2];
ry(-2.6406056506067936) q[3];
cx q[2],q[3];
ry(-0.007207311895877311) q[0];
ry(-2.6939166282905638) q[1];
cx q[0],q[1];
ry(-2.2530476393652332) q[0];
ry(1.3982790955276043) q[1];
cx q[0],q[1];
ry(-2.7707860134711604) q[0];
ry(1.715023361752054) q[2];
cx q[0],q[2];
ry(0.310542605801507) q[0];
ry(-2.1620502568956415) q[2];
cx q[0],q[2];
ry(-2.180552196035817) q[0];
ry(-0.07450446811714696) q[3];
cx q[0],q[3];
ry(-2.1054971799391367) q[0];
ry(1.0857805642994132) q[3];
cx q[0],q[3];
ry(1.5811795066582919) q[1];
ry(-1.4344748843224153) q[2];
cx q[1],q[2];
ry(-1.1315543858355146) q[1];
ry(-1.3906405562093083) q[2];
cx q[1],q[2];
ry(1.8345487378457375) q[1];
ry(2.110888312474599) q[3];
cx q[1],q[3];
ry(2.6466484809807276) q[1];
ry(-1.8581669758779356) q[3];
cx q[1],q[3];
ry(-0.07250037715083936) q[2];
ry(-3.013515571298311) q[3];
cx q[2],q[3];
ry(-1.5828590207661053) q[2];
ry(-1.9613780973321142) q[3];
cx q[2],q[3];
ry(-0.7878666005594573) q[0];
ry(3.1410991669718626) q[1];
cx q[0],q[1];
ry(-1.2346591391639095) q[0];
ry(-1.4139230790695632) q[1];
cx q[0],q[1];
ry(-1.7620134991617782) q[0];
ry(0.6236803315225927) q[2];
cx q[0],q[2];
ry(-0.14614327338940128) q[0];
ry(-2.0323361930517385) q[2];
cx q[0],q[2];
ry(1.0026056318466716) q[0];
ry(-1.3836704199643508) q[3];
cx q[0],q[3];
ry(-0.9969605607528189) q[0];
ry(1.3362541763178697) q[3];
cx q[0],q[3];
ry(-2.162640348088506) q[1];
ry(1.0577665605885893) q[2];
cx q[1],q[2];
ry(0.9905981839049095) q[1];
ry(-0.8741977516280801) q[2];
cx q[1],q[2];
ry(-0.39996049972523284) q[1];
ry(-1.261268583227455) q[3];
cx q[1],q[3];
ry(2.1194273206538323) q[1];
ry(0.40408836794176395) q[3];
cx q[1],q[3];
ry(-1.937177570040313) q[2];
ry(1.3906824341742423) q[3];
cx q[2],q[3];
ry(2.4465974882155783) q[2];
ry(2.1047823129529535) q[3];
cx q[2],q[3];
ry(1.1763145604683536) q[0];
ry(-2.531954213796887) q[1];
cx q[0],q[1];
ry(-0.5202716814172988) q[0];
ry(-1.3695819881038662) q[1];
cx q[0],q[1];
ry(2.268467092403381) q[0];
ry(-2.0635451163731284) q[2];
cx q[0],q[2];
ry(-0.9233383288364267) q[0];
ry(1.0019623402440407) q[2];
cx q[0],q[2];
ry(1.507491473558357) q[0];
ry(-1.276738157225819) q[3];
cx q[0],q[3];
ry(1.7526699174073048) q[0];
ry(0.10946066418068212) q[3];
cx q[0],q[3];
ry(-0.44306170739688344) q[1];
ry(2.8857728468511445) q[2];
cx q[1],q[2];
ry(-1.5582176130827805) q[1];
ry(-2.81178602725854) q[2];
cx q[1],q[2];
ry(-0.8893389977883481) q[1];
ry(-1.9296598324079623) q[3];
cx q[1],q[3];
ry(0.022146244441833496) q[1];
ry(0.5799477775508822) q[3];
cx q[1],q[3];
ry(-0.3504891237897151) q[2];
ry(-1.590489546511693) q[3];
cx q[2],q[3];
ry(-1.7746686274198806) q[2];
ry(1.5910259196247099) q[3];
cx q[2],q[3];
ry(2.003245507126521) q[0];
ry(-2.4399373613514617) q[1];
cx q[0],q[1];
ry(-0.3078705906633576) q[0];
ry(1.9577344429931056) q[1];
cx q[0],q[1];
ry(1.531413568894279) q[0];
ry(0.6912594492850043) q[2];
cx q[0],q[2];
ry(-0.6693299510841488) q[0];
ry(-1.8426769684809132) q[2];
cx q[0],q[2];
ry(-2.762488627305868) q[0];
ry(0.07870243849698255) q[3];
cx q[0],q[3];
ry(0.49266437319729667) q[0];
ry(-2.161930349000805) q[3];
cx q[0],q[3];
ry(1.6847512736531547) q[1];
ry(-2.0741868315988805) q[2];
cx q[1],q[2];
ry(-1.4071697936997074) q[1];
ry(-1.5452134119003986) q[2];
cx q[1],q[2];
ry(-2.223891082341316) q[1];
ry(-2.300291891566407) q[3];
cx q[1],q[3];
ry(-1.9437206809307204) q[1];
ry(-1.6268489222943792) q[3];
cx q[1],q[3];
ry(-1.5931020018437678) q[2];
ry(1.4727932499717011) q[3];
cx q[2],q[3];
ry(2.465785143586853) q[2];
ry(0.8471500738454809) q[3];
cx q[2],q[3];
ry(-0.31288050166261705) q[0];
ry(2.16854529302854) q[1];
cx q[0],q[1];
ry(0.649011156671995) q[0];
ry(-1.4987340363900903) q[1];
cx q[0],q[1];
ry(-2.4656618467567926) q[0];
ry(2.8054820076221545) q[2];
cx q[0],q[2];
ry(-0.2126592156959548) q[0];
ry(2.619602406438217) q[2];
cx q[0],q[2];
ry(-0.20935689474483787) q[0];
ry(1.6229377257057835) q[3];
cx q[0],q[3];
ry(-2.238735332380413) q[0];
ry(-2.433187482581845) q[3];
cx q[0],q[3];
ry(-2.056338627529364) q[1];
ry(1.5025212524976899) q[2];
cx q[1],q[2];
ry(1.1258040014449096) q[1];
ry(2.995491665274697) q[2];
cx q[1],q[2];
ry(-2.367467472931883) q[1];
ry(0.166556776678359) q[3];
cx q[1],q[3];
ry(-1.5224501396693155) q[1];
ry(-2.3394073380839906) q[3];
cx q[1],q[3];
ry(2.9268179257493587) q[2];
ry(1.4203100427119084) q[3];
cx q[2],q[3];
ry(1.1355576048143634) q[2];
ry(2.833720591360551) q[3];
cx q[2],q[3];
ry(2.797581399344648) q[0];
ry(3.0336630044073596) q[1];
cx q[0],q[1];
ry(0.3446621691960277) q[0];
ry(0.727106565071452) q[1];
cx q[0],q[1];
ry(-2.7305523259460096) q[0];
ry(2.2016855003498628) q[2];
cx q[0],q[2];
ry(0.9571702689290227) q[0];
ry(-2.3138120150364094) q[2];
cx q[0],q[2];
ry(-2.063688623382027) q[0];
ry(-0.882020722292635) q[3];
cx q[0],q[3];
ry(0.8259781446875336) q[0];
ry(-0.44863265364692406) q[3];
cx q[0],q[3];
ry(3.058930450492225) q[1];
ry(-1.964828605476548) q[2];
cx q[1],q[2];
ry(-3.135318157751664) q[1];
ry(-2.769272630228356) q[2];
cx q[1],q[2];
ry(-2.5230385072451056) q[1];
ry(-1.5791875100247594) q[3];
cx q[1],q[3];
ry(-0.8854892645860567) q[1];
ry(1.2534653941150111) q[3];
cx q[1],q[3];
ry(-1.4173052313898185) q[2];
ry(1.1478956214198988) q[3];
cx q[2],q[3];
ry(-1.135731411154632) q[2];
ry(-0.7759267903166224) q[3];
cx q[2],q[3];
ry(1.8531751117652577) q[0];
ry(1.376916226773993) q[1];
cx q[0],q[1];
ry(1.7046570843432367) q[0];
ry(2.0842050555241296) q[1];
cx q[0],q[1];
ry(-1.7134529600944934) q[0];
ry(1.3134634016389903) q[2];
cx q[0],q[2];
ry(1.9096709455075693) q[0];
ry(0.6674148931368206) q[2];
cx q[0],q[2];
ry(1.7250085309318524) q[0];
ry(-2.4567952950066663) q[3];
cx q[0],q[3];
ry(-1.1050563157278175) q[0];
ry(0.7670187179637996) q[3];
cx q[0],q[3];
ry(-1.3614068478482553) q[1];
ry(1.5918244140364122) q[2];
cx q[1],q[2];
ry(2.637554498330757) q[1];
ry(2.842478544346763) q[2];
cx q[1],q[2];
ry(1.595650993556123) q[1];
ry(-0.7413195111632644) q[3];
cx q[1],q[3];
ry(0.42739222471312294) q[1];
ry(2.9889282766500407) q[3];
cx q[1],q[3];
ry(-0.1942322584380544) q[2];
ry(-0.7330301243745962) q[3];
cx q[2],q[3];
ry(-2.4295874976324225) q[2];
ry(-0.5421222213923697) q[3];
cx q[2],q[3];
ry(2.6308073915365267) q[0];
ry(-2.8574380538132162) q[1];
cx q[0],q[1];
ry(-1.9735927094762573) q[0];
ry(-0.6734262083915681) q[1];
cx q[0],q[1];
ry(-0.7497359435160337) q[0];
ry(0.015586776672504854) q[2];
cx q[0],q[2];
ry(1.0828606900265854) q[0];
ry(-2.1137707813829882) q[2];
cx q[0],q[2];
ry(-2.126244190032909) q[0];
ry(1.170291155064925) q[3];
cx q[0],q[3];
ry(2.9633318810589575) q[0];
ry(2.7394571053832197) q[3];
cx q[0],q[3];
ry(-2.959155697238958) q[1];
ry(-0.11957514788848835) q[2];
cx q[1],q[2];
ry(-0.4014531483209018) q[1];
ry(1.6762071586119722) q[2];
cx q[1],q[2];
ry(-2.653936429696614) q[1];
ry(-1.5665101872466431) q[3];
cx q[1],q[3];
ry(0.6180135144009619) q[1];
ry(-0.9051995031473087) q[3];
cx q[1],q[3];
ry(0.26098162975140965) q[2];
ry(1.7707683432596282) q[3];
cx q[2],q[3];
ry(-0.7277434178857733) q[2];
ry(-2.9019557286513873) q[3];
cx q[2],q[3];
ry(-2.4375167958278428) q[0];
ry(1.1017889148706996) q[1];
cx q[0],q[1];
ry(-0.8440843059985036) q[0];
ry(0.9484601227423175) q[1];
cx q[0],q[1];
ry(-0.8906278381119012) q[0];
ry(0.15347773273347148) q[2];
cx q[0],q[2];
ry(-1.6637730068685197) q[0];
ry(-0.48212771876183114) q[2];
cx q[0],q[2];
ry(-0.5105761623514029) q[0];
ry(1.5644410804465494) q[3];
cx q[0],q[3];
ry(-1.1624302639635657) q[0];
ry(-0.6303929469661211) q[3];
cx q[0],q[3];
ry(-0.47640328885938715) q[1];
ry(-1.452591879551186) q[2];
cx q[1],q[2];
ry(0.6620243127983088) q[1];
ry(-1.620218580447436) q[2];
cx q[1],q[2];
ry(-2.859380025562988) q[1];
ry(2.9051405455537282) q[3];
cx q[1],q[3];
ry(2.050540566014389) q[1];
ry(-0.44543186248508976) q[3];
cx q[1],q[3];
ry(-2.963119829952707) q[2];
ry(1.841998993747401) q[3];
cx q[2],q[3];
ry(0.2055678367660732) q[2];
ry(-0.9330748286909651) q[3];
cx q[2],q[3];
ry(-0.1757491233518894) q[0];
ry(-0.17265360354787518) q[1];
cx q[0],q[1];
ry(-1.8405650626857721) q[0];
ry(-0.7704566588138885) q[1];
cx q[0],q[1];
ry(-2.703399893481373) q[0];
ry(2.368228259135585) q[2];
cx q[0],q[2];
ry(-0.8794575466705794) q[0];
ry(1.8442450686247138) q[2];
cx q[0],q[2];
ry(0.24737486476246417) q[0];
ry(-1.4125190319832122) q[3];
cx q[0],q[3];
ry(-2.272044250693308) q[0];
ry(-2.714285569444506) q[3];
cx q[0],q[3];
ry(2.1797909014458225) q[1];
ry(-3.037542677424755) q[2];
cx q[1],q[2];
ry(0.6936948338202474) q[1];
ry(1.3986686254520349) q[2];
cx q[1],q[2];
ry(-1.2695961695268805) q[1];
ry(-1.9197401747063934) q[3];
cx q[1],q[3];
ry(-2.1570330697981834) q[1];
ry(1.968857242124555) q[3];
cx q[1],q[3];
ry(-1.9387700836314268) q[2];
ry(2.3994110788343566) q[3];
cx q[2],q[3];
ry(-1.0379975711515192) q[2];
ry(2.2663784961142506) q[3];
cx q[2],q[3];
ry(-2.831441309116681) q[0];
ry(-0.6240670123124349) q[1];
cx q[0],q[1];
ry(-2.7040279109078553) q[0];
ry(2.2471815758114246) q[1];
cx q[0],q[1];
ry(1.435026319721321) q[0];
ry(-0.7989857688427611) q[2];
cx q[0],q[2];
ry(1.3460755053056728) q[0];
ry(0.06394966643556454) q[2];
cx q[0],q[2];
ry(2.0951461648799174) q[0];
ry(-1.3929479433709728) q[3];
cx q[0],q[3];
ry(0.9796117996575838) q[0];
ry(1.1448820016577919) q[3];
cx q[0],q[3];
ry(0.6917874885225093) q[1];
ry(3.0343122827294606) q[2];
cx q[1],q[2];
ry(2.2165151343826173) q[1];
ry(-0.5848179452099744) q[2];
cx q[1],q[2];
ry(1.0526820461292654) q[1];
ry(-1.080223646195785) q[3];
cx q[1],q[3];
ry(1.8074071176547217) q[1];
ry(-0.0966965468884413) q[3];
cx q[1],q[3];
ry(0.11113028714533701) q[2];
ry(-2.867681760695279) q[3];
cx q[2],q[3];
ry(-3.0240200243736317) q[2];
ry(2.260377437320093) q[3];
cx q[2],q[3];
ry(-2.734773208211024) q[0];
ry(1.5893645596660333) q[1];
cx q[0],q[1];
ry(-1.6215197946935178) q[0];
ry(-1.4983436904624137) q[1];
cx q[0],q[1];
ry(-2.462826996950761) q[0];
ry(-0.7702155274935901) q[2];
cx q[0],q[2];
ry(-1.0352023288424872) q[0];
ry(-2.3989616307475674) q[2];
cx q[0],q[2];
ry(-2.2758508339105985) q[0];
ry(0.9950198469236298) q[3];
cx q[0],q[3];
ry(2.0694218535777416) q[0];
ry(0.9364498309412469) q[3];
cx q[0],q[3];
ry(2.309376864031905) q[1];
ry(1.2940225461593138) q[2];
cx q[1],q[2];
ry(-2.285559149584706) q[1];
ry(-1.488973523255619) q[2];
cx q[1],q[2];
ry(1.048170469250044) q[1];
ry(1.027838330497128) q[3];
cx q[1],q[3];
ry(1.6813497154573973) q[1];
ry(2.603585251518686) q[3];
cx q[1],q[3];
ry(0.7897408551227543) q[2];
ry(2.9854385785252124) q[3];
cx q[2],q[3];
ry(-2.4059824891527537) q[2];
ry(-2.6471060661964647) q[3];
cx q[2],q[3];
ry(0.09701357824882706) q[0];
ry(0.08270422437839288) q[1];
cx q[0],q[1];
ry(1.3447591881939849) q[0];
ry(0.10771602208993336) q[1];
cx q[0],q[1];
ry(-1.20298209015252) q[0];
ry(-1.8923114675860446) q[2];
cx q[0],q[2];
ry(-2.7596803486006976) q[0];
ry(-2.752856079090895) q[2];
cx q[0],q[2];
ry(1.0449138801239055) q[0];
ry(2.474105045098167) q[3];
cx q[0],q[3];
ry(0.6511569117455835) q[0];
ry(2.3481533868208206) q[3];
cx q[0],q[3];
ry(0.20942563232638908) q[1];
ry(2.4539451338301297) q[2];
cx q[1],q[2];
ry(-1.0675303419859612) q[1];
ry(-0.7220899690661624) q[2];
cx q[1],q[2];
ry(-3.015146140442834) q[1];
ry(-0.18672420703179426) q[3];
cx q[1],q[3];
ry(1.551390880043189) q[1];
ry(2.7831714894581503) q[3];
cx q[1],q[3];
ry(-2.919075476520643) q[2];
ry(1.7263784699857836) q[3];
cx q[2],q[3];
ry(0.05398885451804514) q[2];
ry(0.6316624117956566) q[3];
cx q[2],q[3];
ry(0.6732045332297362) q[0];
ry(1.0016540620174772) q[1];
ry(-2.5448875724953153) q[2];
ry(-1.7628457056098084) q[3];