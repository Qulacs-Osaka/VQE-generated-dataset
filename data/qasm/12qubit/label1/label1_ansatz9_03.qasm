OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.2774537642827033) q[0];
ry(-2.41204885146806) q[1];
cx q[0],q[1];
ry(-2.5412496962623234) q[0];
ry(-1.2187454283853782) q[1];
cx q[0],q[1];
ry(2.4088086603295933) q[2];
ry(0.9331406273016096) q[3];
cx q[2],q[3];
ry(-0.4110577504309756) q[2];
ry(-1.762029422015347) q[3];
cx q[2],q[3];
ry(0.4129355417649939) q[4];
ry(1.2786971901832276) q[5];
cx q[4],q[5];
ry(-2.223609699713154) q[4];
ry(0.2715326750168204) q[5];
cx q[4],q[5];
ry(-2.657443600639493) q[6];
ry(-1.3629091692294386) q[7];
cx q[6],q[7];
ry(-0.1548848927555584) q[6];
ry(-0.3052581966390031) q[7];
cx q[6],q[7];
ry(-2.395556497237791) q[8];
ry(-1.966190565349434) q[9];
cx q[8],q[9];
ry(-0.8711017290423075) q[8];
ry(2.637986136008563) q[9];
cx q[8],q[9];
ry(-2.052990484412109) q[10];
ry(-1.7600131989273124) q[11];
cx q[10],q[11];
ry(-1.3249684245021918) q[10];
ry(-2.1087768975590127) q[11];
cx q[10],q[11];
ry(-0.9553177818491916) q[0];
ry(1.9007747701154738) q[2];
cx q[0],q[2];
ry(-0.42984287529504533) q[0];
ry(-2.1792152501670827) q[2];
cx q[0],q[2];
ry(2.0441661783547582) q[2];
ry(0.17731431228147138) q[4];
cx q[2],q[4];
ry(-3.1297386547065034) q[2];
ry(-0.0005844763734828007) q[4];
cx q[2],q[4];
ry(-0.712685920776333) q[4];
ry(-2.2857582613688785) q[6];
cx q[4],q[6];
ry(-2.2274467469506654) q[4];
ry(0.5760044216471211) q[6];
cx q[4],q[6];
ry(0.4897443874625057) q[6];
ry(0.41274103820554064) q[8];
cx q[6],q[8];
ry(1.6418251938586463) q[6];
ry(2.308052220717399) q[8];
cx q[6],q[8];
ry(-0.17071364101269595) q[8];
ry(2.616050351866216) q[10];
cx q[8],q[10];
ry(-0.7629070614624816) q[8];
ry(1.5150774723724767) q[10];
cx q[8],q[10];
ry(-1.2438606041365412) q[1];
ry(-1.2931837485228543) q[3];
cx q[1],q[3];
ry(-2.6349767142408274) q[1];
ry(-1.270555453512397) q[3];
cx q[1],q[3];
ry(-0.5699132083381738) q[3];
ry(-2.5283352339038405) q[5];
cx q[3],q[5];
ry(-0.009581317149369299) q[3];
ry(3.1261644075050437) q[5];
cx q[3],q[5];
ry(2.46548292184542) q[5];
ry(0.1517183922264005) q[7];
cx q[5],q[7];
ry(2.9445188474622723) q[5];
ry(-3.0134644112732984) q[7];
cx q[5],q[7];
ry(-1.1685346510787857) q[7];
ry(-1.7532255713919307) q[9];
cx q[7],q[9];
ry(-2.0938568391156505) q[7];
ry(2.1844528143517064) q[9];
cx q[7],q[9];
ry(2.823673239851672) q[9];
ry(-0.21839258555108113) q[11];
cx q[9],q[11];
ry(1.7223157936675362) q[9];
ry(-1.4007664101819701) q[11];
cx q[9],q[11];
ry(-0.633538581678855) q[0];
ry(-0.6070404337062189) q[3];
cx q[0],q[3];
ry(-1.3990412566979267) q[0];
ry(-0.6565598667972222) q[3];
cx q[0],q[3];
ry(0.3816925999846328) q[1];
ry(0.9523513455713427) q[2];
cx q[1],q[2];
ry(2.3287069238036877) q[1];
ry(0.632143574893707) q[2];
cx q[1],q[2];
ry(1.565304258140154) q[2];
ry(-1.3135077817124552) q[5];
cx q[2],q[5];
ry(0.009280519809880552) q[2];
ry(-0.0016039511906249587) q[5];
cx q[2],q[5];
ry(1.5077420191562272) q[3];
ry(1.3590267570775074) q[4];
cx q[3],q[4];
ry(-0.0004823332362331456) q[3];
ry(0.0006541558866395292) q[4];
cx q[3],q[4];
ry(1.4804544092606085) q[4];
ry(1.9100197721405767) q[7];
cx q[4],q[7];
ry(1.2802689319231861) q[4];
ry(1.784892823041096) q[7];
cx q[4],q[7];
ry(0.5019566893126717) q[5];
ry(-1.981877863986674) q[6];
cx q[5],q[6];
ry(-0.0005562979104398735) q[5];
ry(0.044689371302051306) q[6];
cx q[5],q[6];
ry(0.675757305935904) q[6];
ry(-0.16161928227931366) q[9];
cx q[6],q[9];
ry(-0.5758690542474686) q[6];
ry(-2.2216782585917736) q[9];
cx q[6],q[9];
ry(-0.14295722478953277) q[7];
ry(2.8783876154835917) q[8];
cx q[7],q[8];
ry(1.2119893546481528) q[7];
ry(-1.8048145804762272) q[8];
cx q[7],q[8];
ry(-2.0897791153365324) q[8];
ry(3.131923518591459) q[11];
cx q[8],q[11];
ry(2.686672957563444) q[8];
ry(-2.930276730668006) q[11];
cx q[8],q[11];
ry(-3.1089146928593703) q[9];
ry(1.5313314438971566) q[10];
cx q[9],q[10];
ry(-2.0193097332222587) q[9];
ry(-1.3761040826712618) q[10];
cx q[9],q[10];
ry(-2.8264696232030366) q[0];
ry(1.3610688876709922) q[1];
cx q[0],q[1];
ry(-0.15503986059077643) q[0];
ry(-0.9279618483701526) q[1];
cx q[0],q[1];
ry(-1.797199446839243) q[2];
ry(1.873368303595476) q[3];
cx q[2],q[3];
ry(-2.1061573875162813) q[2];
ry(-0.3195641134908458) q[3];
cx q[2],q[3];
ry(-1.573658577912821) q[4];
ry(2.4193158202631095) q[5];
cx q[4],q[5];
ry(-0.2149693691717989) q[4];
ry(-1.074869415733846) q[5];
cx q[4],q[5];
ry(-0.9885680247323468) q[6];
ry(1.3304379716113586) q[7];
cx q[6],q[7];
ry(-2.7842121030900944) q[6];
ry(-2.239131652831388) q[7];
cx q[6],q[7];
ry(2.6141607387577066) q[8];
ry(-2.825700764415993) q[9];
cx q[8],q[9];
ry(0.31108224244636196) q[8];
ry(-1.6436396830136266) q[9];
cx q[8],q[9];
ry(-0.36529181716471015) q[10];
ry(0.5099796529520693) q[11];
cx q[10],q[11];
ry(1.9466660945099) q[10];
ry(-2.636162843141761) q[11];
cx q[10],q[11];
ry(-0.419795561767456) q[0];
ry(-0.4634839905095927) q[2];
cx q[0],q[2];
ry(0.8050345707479998) q[0];
ry(2.355324766638538) q[2];
cx q[0],q[2];
ry(-1.3341303029057172) q[2];
ry(-2.584648636859558) q[4];
cx q[2],q[4];
ry(3.138944162895407) q[2];
ry(3.139354591372467) q[4];
cx q[2],q[4];
ry(0.046102589399548234) q[4];
ry(3.1357246682175446) q[6];
cx q[4],q[6];
ry(-3.141047239964469) q[4];
ry(-3.1270012890977212) q[6];
cx q[4],q[6];
ry(-1.9929033042003041) q[6];
ry(-2.0642686539221153) q[8];
cx q[6],q[8];
ry(-1.2748634668056127) q[6];
ry(1.32170667327902) q[8];
cx q[6],q[8];
ry(3.0289138530710114) q[8];
ry(2.790979920093275) q[10];
cx q[8],q[10];
ry(2.6362628834621185) q[8];
ry(2.5773921755487015) q[10];
cx q[8],q[10];
ry(0.38950985621344314) q[1];
ry(2.349191269711994) q[3];
cx q[1],q[3];
ry(-3.0298186626626578) q[1];
ry(2.2776694065352756) q[3];
cx q[1],q[3];
ry(-0.4264892842795151) q[3];
ry(2.838955766545641) q[5];
cx q[3],q[5];
ry(-0.004364555175091489) q[3];
ry(-3.135490208784773) q[5];
cx q[3],q[5];
ry(-0.3643976842259767) q[5];
ry(0.6635384806805563) q[7];
cx q[5],q[7];
ry(-0.07402948597487045) q[5];
ry(-3.02417418697879) q[7];
cx q[5],q[7];
ry(0.0007263339792690682) q[7];
ry(0.3514939910557917) q[9];
cx q[7],q[9];
ry(-1.389923100689131) q[7];
ry(-2.496053512712976) q[9];
cx q[7],q[9];
ry(1.5230576895689607) q[9];
ry(1.4508872385434095) q[11];
cx q[9],q[11];
ry(2.663336805386844) q[9];
ry(1.701948826446513) q[11];
cx q[9],q[11];
ry(1.4718691055563617) q[0];
ry(-2.1839122729459426) q[3];
cx q[0],q[3];
ry(2.759807522898024) q[0];
ry(0.5740698514085493) q[3];
cx q[0],q[3];
ry(-0.4106118505619813) q[1];
ry(0.02406877535297838) q[2];
cx q[1],q[2];
ry(-3.0003445404938622) q[1];
ry(0.03291031446019888) q[2];
cx q[1],q[2];
ry(2.3445200509755053) q[2];
ry(0.549508680179545) q[5];
cx q[2],q[5];
ry(-0.0264884315572723) q[2];
ry(3.1394409510924466) q[5];
cx q[2],q[5];
ry(0.5205971019251516) q[3];
ry(1.5810905030492821) q[4];
cx q[3],q[4];
ry(-3.1305933039290315) q[3];
ry(3.139084647339617) q[4];
cx q[3],q[4];
ry(-1.224076873076422) q[4];
ry(0.5709780691461593) q[7];
cx q[4],q[7];
ry(-3.1338942608140203) q[4];
ry(3.126439717188786) q[7];
cx q[4],q[7];
ry(1.189990772274288) q[5];
ry(2.3270230950086885) q[6];
cx q[5],q[6];
ry(-3.135891970636046) q[5];
ry(3.114588691327991) q[6];
cx q[5],q[6];
ry(-0.8169376167270208) q[6];
ry(1.1013357489227644) q[9];
cx q[6],q[9];
ry(-0.8756505282899143) q[6];
ry(0.7340924491361441) q[9];
cx q[6],q[9];
ry(-1.0471841024489148) q[7];
ry(-2.531256422814765) q[8];
cx q[7],q[8];
ry(1.7343759734783948) q[7];
ry(-2.8025993276074592) q[8];
cx q[7],q[8];
ry(-0.32179291057059256) q[8];
ry(-1.2177756232016996) q[11];
cx q[8],q[11];
ry(1.6435366597275953) q[8];
ry(-0.2418570049456088) q[11];
cx q[8],q[11];
ry(-2.211736281737292) q[9];
ry(0.15458735880067034) q[10];
cx q[9],q[10];
ry(0.7768553497496726) q[9];
ry(2.699046047612096) q[10];
cx q[9],q[10];
ry(1.7565881238035557) q[0];
ry(2.2040095181128097) q[1];
cx q[0],q[1];
ry(-3.0867949501192786) q[0];
ry(2.1001593618562753) q[1];
cx q[0],q[1];
ry(0.46692602623550883) q[2];
ry(0.7735142342382434) q[3];
cx q[2],q[3];
ry(-1.5365531400156376) q[2];
ry(0.6165670062533666) q[3];
cx q[2],q[3];
ry(0.20591659714338295) q[4];
ry(-0.05592230412390098) q[5];
cx q[4],q[5];
ry(2.741617837386123) q[4];
ry(1.508533265916524) q[5];
cx q[4],q[5];
ry(-0.2532016491057383) q[6];
ry(1.0163068377169944) q[7];
cx q[6],q[7];
ry(-0.6821796399123479) q[6];
ry(-2.6034554169492887) q[7];
cx q[6],q[7];
ry(-2.508888195379465) q[8];
ry(0.43070858528711703) q[9];
cx q[8],q[9];
ry(2.0923321413636984) q[8];
ry(0.7557784662120068) q[9];
cx q[8],q[9];
ry(-0.5187622080191918) q[10];
ry(-0.3540079190149754) q[11];
cx q[10],q[11];
ry(1.443155989986591) q[10];
ry(0.6963404972612492) q[11];
cx q[10],q[11];
ry(-0.8371019574279978) q[0];
ry(1.4271108873212284) q[2];
cx q[0],q[2];
ry(2.3210259528372137) q[0];
ry(0.903131562597851) q[2];
cx q[0],q[2];
ry(0.9338866797350384) q[2];
ry(-1.0780716371114885) q[4];
cx q[2],q[4];
ry(-3.025305781217054) q[2];
ry(-0.02614184052833135) q[4];
cx q[2],q[4];
ry(1.692665637072933) q[4];
ry(-0.8785194897911136) q[6];
cx q[4],q[6];
ry(-3.131560106189252) q[4];
ry(0.0777395081406711) q[6];
cx q[4],q[6];
ry(-1.0473216193004573) q[6];
ry(-1.1227252959674578) q[8];
cx q[6],q[8];
ry(0.837731925151142) q[6];
ry(1.7411904770953566) q[8];
cx q[6],q[8];
ry(2.468336537575636) q[8];
ry(-1.8994529095428094) q[10];
cx q[8],q[10];
ry(0.7344195188017197) q[8];
ry(2.047346751789065) q[10];
cx q[8],q[10];
ry(-2.9500339174634265) q[1];
ry(2.8172771583608274) q[3];
cx q[1],q[3];
ry(-1.1429647416927073) q[1];
ry(1.992051955123091) q[3];
cx q[1],q[3];
ry(-2.885596653763419) q[3];
ry(1.175587692206766) q[5];
cx q[3],q[5];
ry(-3.111428556474645) q[3];
ry(-0.0033063828466266543) q[5];
cx q[3],q[5];
ry(-1.030162165711473) q[5];
ry(-2.1263529223631608) q[7];
cx q[5],q[7];
ry(-0.018946826296697417) q[5];
ry(0.2549280979007404) q[7];
cx q[5],q[7];
ry(-1.5349606555067012) q[7];
ry(0.7800373275302682) q[9];
cx q[7],q[9];
ry(0.347430734080279) q[7];
ry(0.11334465714636992) q[9];
cx q[7],q[9];
ry(1.5786976048560806) q[9];
ry(-1.616626167295657) q[11];
cx q[9],q[11];
ry(-1.9636965698563564) q[9];
ry(-0.019817521707503616) q[11];
cx q[9],q[11];
ry(1.4836887941152366) q[0];
ry(-1.7358915283046064) q[3];
cx q[0],q[3];
ry(-1.0631263541737752) q[0];
ry(1.1380525545248332) q[3];
cx q[0],q[3];
ry(-0.14339315024409738) q[1];
ry(3.127056580510126) q[2];
cx q[1],q[2];
ry(3.104084875064446) q[1];
ry(0.4299521303457743) q[2];
cx q[1],q[2];
ry(-0.02334338684666386) q[2];
ry(0.5358071646879783) q[5];
cx q[2],q[5];
ry(3.091458779938361) q[2];
ry(-0.012199555011926044) q[5];
cx q[2],q[5];
ry(1.8290071709637565) q[3];
ry(-1.7222649576096034) q[4];
cx q[3],q[4];
ry(-0.04557954848503698) q[3];
ry(-0.021885496650577707) q[4];
cx q[3],q[4];
ry(1.5366228774176072) q[4];
ry(-1.9236582619568592) q[7];
cx q[4],q[7];
ry(3.1345591124245193) q[4];
ry(-0.2241014880178609) q[7];
cx q[4],q[7];
ry(1.0893752368718204) q[5];
ry(-2.6981792949801617) q[6];
cx q[5],q[6];
ry(-3.1209890636458275) q[5];
ry(-3.0531119369045268) q[6];
cx q[5],q[6];
ry(-2.462687137607965) q[6];
ry(-0.7471519665957205) q[9];
cx q[6],q[9];
ry(0.7800519232496971) q[6];
ry(-2.154086966006994) q[9];
cx q[6],q[9];
ry(1.9957371919534914) q[7];
ry(-3.138455155991757) q[8];
cx q[7],q[8];
ry(-1.3485957611435753) q[7];
ry(-0.1076450122210394) q[8];
cx q[7],q[8];
ry(-3.1249909908823845) q[8];
ry(1.805416000594115) q[11];
cx q[8],q[11];
ry(-1.9735473513418214) q[8];
ry(-1.4938306173414757) q[11];
cx q[8],q[11];
ry(1.0159933070314677) q[9];
ry(1.7298332712144755) q[10];
cx q[9],q[10];
ry(0.45361405814839345) q[9];
ry(-0.7686192224621395) q[10];
cx q[9],q[10];
ry(-1.9133323547890333) q[0];
ry(0.01873505286071797) q[1];
cx q[0],q[1];
ry(0.9025721423018983) q[0];
ry(-1.78476675945503) q[1];
cx q[0],q[1];
ry(-1.2856758881797792) q[2];
ry(-1.2396513837793206) q[3];
cx q[2],q[3];
ry(0.8093692943261921) q[2];
ry(-2.5466128002148705) q[3];
cx q[2],q[3];
ry(2.226640270561457) q[4];
ry(-1.7848514853559208) q[5];
cx q[4],q[5];
ry(1.7122017788306472) q[4];
ry(-2.907400065421839) q[5];
cx q[4],q[5];
ry(3.1042023241562897) q[6];
ry(-2.010038101914726) q[7];
cx q[6],q[7];
ry(-2.988395707762139) q[6];
ry(2.94426415874001) q[7];
cx q[6],q[7];
ry(-1.4807405671252338) q[8];
ry(-0.9251435904193804) q[9];
cx q[8],q[9];
ry(-0.5886076721832602) q[8];
ry(2.37644751920574) q[9];
cx q[8],q[9];
ry(-1.8963319850263567) q[10];
ry(2.225645846847596) q[11];
cx q[10],q[11];
ry(2.8939199269974933) q[10];
ry(1.8472813214949673) q[11];
cx q[10],q[11];
ry(1.0619579349709252) q[0];
ry(-0.7790323293855659) q[2];
cx q[0],q[2];
ry(0.29738516610730614) q[0];
ry(-2.9593095280336796) q[2];
cx q[0],q[2];
ry(-0.7209123034757936) q[2];
ry(2.0239325479549146) q[4];
cx q[2],q[4];
ry(2.9588704710646736) q[2];
ry(-3.0599638704655283) q[4];
cx q[2],q[4];
ry(-1.4825823407907286) q[4];
ry(1.102828466995487) q[6];
cx q[4],q[6];
ry(0.0039059362893123506) q[4];
ry(3.1064144382254146) q[6];
cx q[4],q[6];
ry(-0.773736913935469) q[6];
ry(2.261799768143563) q[8];
cx q[6],q[8];
ry(-0.9877561066480106) q[6];
ry(-2.493794489572139) q[8];
cx q[6],q[8];
ry(1.4819473729659949) q[8];
ry(-2.445759623278426) q[10];
cx q[8],q[10];
ry(-0.4541613379108602) q[8];
ry(-2.231951116909741) q[10];
cx q[8],q[10];
ry(-2.2121116411982147) q[1];
ry(-0.35543260812831645) q[3];
cx q[1],q[3];
ry(-1.0696396727269086) q[1];
ry(0.2959883386605284) q[3];
cx q[1],q[3];
ry(2.8517584117427885) q[3];
ry(-2.9103535827247997) q[5];
cx q[3],q[5];
ry(0.7741293135907732) q[3];
ry(-0.02220936107175664) q[5];
cx q[3],q[5];
ry(1.4806017355460126) q[5];
ry(-2.9920148735157897) q[7];
cx q[5],q[7];
ry(0.0009169073739746346) q[5];
ry(1.0039017521953673) q[7];
cx q[5],q[7];
ry(-0.7499618076005258) q[7];
ry(-1.8329443569976567) q[9];
cx q[7],q[9];
ry(-2.3741399630217788) q[7];
ry(-2.7953117945974557) q[9];
cx q[7],q[9];
ry(0.992437807450522) q[9];
ry(-1.248261984060135) q[11];
cx q[9],q[11];
ry(2.330103573364305) q[9];
ry(2.5896281724489345) q[11];
cx q[9],q[11];
ry(2.7813990850955186) q[0];
ry(-2.9578314689769867) q[3];
cx q[0],q[3];
ry(-2.861792181510178) q[0];
ry(0.3901179744162456) q[3];
cx q[0],q[3];
ry(-1.6415707416297285) q[1];
ry(1.9550736964379498) q[2];
cx q[1],q[2];
ry(-0.15701999776582057) q[1];
ry(0.21278394648923446) q[2];
cx q[1],q[2];
ry(-2.409784431161786) q[2];
ry(1.7119711617855637) q[5];
cx q[2],q[5];
ry(-2.916533020320508) q[2];
ry(0.01842942274639459) q[5];
cx q[2],q[5];
ry(-2.40985981738651) q[3];
ry(-1.3566381318565037) q[4];
cx q[3],q[4];
ry(2.508239521894889) q[3];
ry(-0.06090822525349859) q[4];
cx q[3],q[4];
ry(-0.5696853860276464) q[4];
ry(-2.2838331394857723) q[7];
cx q[4],q[7];
ry(0.002058956338794715) q[4];
ry(-0.005628968536574505) q[7];
cx q[4],q[7];
ry(-1.4681233657125157) q[5];
ry(1.6339577352495442) q[6];
cx q[5],q[6];
ry(-3.1408619874203505) q[5];
ry(-3.09095136679126) q[6];
cx q[5],q[6];
ry(-2.236185719723679) q[6];
ry(1.0683548113249275) q[9];
cx q[6],q[9];
ry(1.2881011519652654) q[6];
ry(-2.5207685843119325) q[9];
cx q[6],q[9];
ry(-0.3489629642190065) q[7];
ry(-2.906678599340731) q[8];
cx q[7],q[8];
ry(-1.1020458774338822) q[7];
ry(2.7395320854085745) q[8];
cx q[7],q[8];
ry(2.175651186323244) q[8];
ry(1.481650362403335) q[11];
cx q[8],q[11];
ry(-2.278064998394993) q[8];
ry(2.9051430929850124) q[11];
cx q[8],q[11];
ry(2.988525634469204) q[9];
ry(2.709900876463893) q[10];
cx q[9],q[10];
ry(1.3445776846453583) q[9];
ry(1.203049053909663) q[10];
cx q[9],q[10];
ry(-2.4621812247893007) q[0];
ry(-0.2852311018622424) q[1];
cx q[0],q[1];
ry(1.5275553800422388) q[0];
ry(1.1330285555453838) q[1];
cx q[0],q[1];
ry(0.7724757834062572) q[2];
ry(2.035058426975964) q[3];
cx q[2],q[3];
ry(3.1387810135075473) q[2];
ry(-0.07693643025210097) q[3];
cx q[2],q[3];
ry(0.6533121180419345) q[4];
ry(1.9084758535373672) q[5];
cx q[4],q[5];
ry(-3.0654804829007367) q[4];
ry(-1.593158498274101) q[5];
cx q[4],q[5];
ry(0.14619448656973777) q[6];
ry(-2.7735263337040044) q[7];
cx q[6],q[7];
ry(-0.22756554938370677) q[6];
ry(2.906092917559696) q[7];
cx q[6],q[7];
ry(-2.3850242399990353) q[8];
ry(-0.4086720710813152) q[9];
cx q[8],q[9];
ry(-1.8835327674175275) q[8];
ry(-0.1386680614240955) q[9];
cx q[8],q[9];
ry(-1.7741497102932486) q[10];
ry(1.8812286582394206) q[11];
cx q[10],q[11];
ry(0.813676596681797) q[10];
ry(-1.4395509909981485) q[11];
cx q[10],q[11];
ry(1.507777478954564) q[0];
ry(1.4041126285690222) q[2];
cx q[0],q[2];
ry(-3.0216422303321053) q[0];
ry(-1.9755539163645395) q[2];
cx q[0],q[2];
ry(-1.5138730285836925) q[2];
ry(0.744061041515245) q[4];
cx q[2],q[4];
ry(-3.138272303396868) q[2];
ry(3.062593985578632) q[4];
cx q[2],q[4];
ry(-1.0393036329980925) q[4];
ry(1.2733690335297636) q[6];
cx q[4],q[6];
ry(-3.141438658137088) q[4];
ry(-3.138289987483839) q[6];
cx q[4],q[6];
ry(-0.2057107176964248) q[6];
ry(-2.66342307040391) q[8];
cx q[6],q[8];
ry(-2.5147900607970666) q[6];
ry(-0.6197154431053206) q[8];
cx q[6],q[8];
ry(2.945985931324991) q[8];
ry(-1.8120659659574017) q[10];
cx q[8],q[10];
ry(0.4654284649885078) q[8];
ry(-3.0850347850661834) q[10];
cx q[8],q[10];
ry(0.4849853551114087) q[1];
ry(-1.8885564906905383) q[3];
cx q[1],q[3];
ry(-0.556506560465062) q[1];
ry(-1.5767525957264283) q[3];
cx q[1],q[3];
ry(1.849120108236817) q[3];
ry(-0.6206033554736963) q[5];
cx q[3],q[5];
ry(0.009588057516535955) q[3];
ry(1.6298651560702855) q[5];
cx q[3],q[5];
ry(-2.227641680745177) q[5];
ry(0.6016620153360006) q[7];
cx q[5],q[7];
ry(3.1340771696728296) q[5];
ry(0.27556535358628464) q[7];
cx q[5],q[7];
ry(2.021373335435676) q[7];
ry(-1.0143760128026527) q[9];
cx q[7],q[9];
ry(-0.07825881874301999) q[7];
ry(-3.103807087873466) q[9];
cx q[7],q[9];
ry(-0.9256006392985856) q[9];
ry(-1.4310477654771294) q[11];
cx q[9],q[11];
ry(-0.4875152201942248) q[9];
ry(0.892026209711636) q[11];
cx q[9],q[11];
ry(2.6327869060478526) q[0];
ry(1.392462850734268) q[3];
cx q[0],q[3];
ry(0.5404582002577918) q[0];
ry(-1.943818353208644) q[3];
cx q[0],q[3];
ry(-0.9497505275856436) q[1];
ry(-2.1695742883568228) q[2];
cx q[1],q[2];
ry(-1.9375996712168) q[1];
ry(-1.2794941402525086) q[2];
cx q[1],q[2];
ry(2.3286358379222616) q[2];
ry(2.00454382016089) q[5];
cx q[2],q[5];
ry(-0.009038292390154188) q[2];
ry(-3.134578477889386) q[5];
cx q[2],q[5];
ry(3.0938025693154487) q[3];
ry(-2.554071975658531) q[4];
cx q[3],q[4];
ry(0.014008560374562131) q[3];
ry(0.026547130240701083) q[4];
cx q[3],q[4];
ry(-1.0223623710647474) q[4];
ry(2.702258164047055) q[7];
cx q[4],q[7];
ry(0.002866285618660249) q[4];
ry(0.03827821237520013) q[7];
cx q[4],q[7];
ry(1.3285369155385947) q[5];
ry(-0.09042549197016025) q[6];
cx q[5],q[6];
ry(3.136764685576296) q[5];
ry(3.13598452141003) q[6];
cx q[5],q[6];
ry(1.410077533919403) q[6];
ry(-0.9686704756106703) q[9];
cx q[6],q[9];
ry(0.8197004576189668) q[6];
ry(1.4800273529353276) q[9];
cx q[6],q[9];
ry(1.596247237403067) q[7];
ry(0.706816164181598) q[8];
cx q[7],q[8];
ry(-1.7073162079441342) q[7];
ry(-0.24399921964927443) q[8];
cx q[7],q[8];
ry(2.9789123735125576) q[8];
ry(2.4787363346662237) q[11];
cx q[8],q[11];
ry(-2.27420846050582) q[8];
ry(2.7626918071421844) q[11];
cx q[8],q[11];
ry(2.649420616320442) q[9];
ry(-1.414067614024931) q[10];
cx q[9],q[10];
ry(3.0998856299450943) q[9];
ry(-3.026470432672281) q[10];
cx q[9],q[10];
ry(-1.8092797543566423) q[0];
ry(1.67161686283767) q[1];
cx q[0],q[1];
ry(-2.4598886090329692) q[0];
ry(-2.81474075778975) q[1];
cx q[0],q[1];
ry(-0.7090919614350133) q[2];
ry(-1.0142230343549725) q[3];
cx q[2],q[3];
ry(2.923199240889954) q[2];
ry(-1.6209690416636755) q[3];
cx q[2],q[3];
ry(0.37138234269901843) q[4];
ry(-1.3658031767796244) q[5];
cx q[4],q[5];
ry(-1.6141814429246626) q[4];
ry(-3.084643941764879) q[5];
cx q[4],q[5];
ry(-2.9996695608413946) q[6];
ry(3.1370233833289336) q[7];
cx q[6],q[7];
ry(-0.7405742833737664) q[6];
ry(1.2347091856595833) q[7];
cx q[6],q[7];
ry(-1.514851828686922) q[8];
ry(-0.8910891227786868) q[9];
cx q[8],q[9];
ry(-3.103696768755687) q[8];
ry(-3.0608565048121865) q[9];
cx q[8],q[9];
ry(0.9548951410090118) q[10];
ry(-0.5483145656966393) q[11];
cx q[10],q[11];
ry(-1.676855000727582) q[10];
ry(0.7112014330498839) q[11];
cx q[10],q[11];
ry(-0.2624646410698716) q[0];
ry(0.851845750021301) q[2];
cx q[0],q[2];
ry(-2.993434278173086) q[0];
ry(-3.061227881731543) q[2];
cx q[0],q[2];
ry(0.699289796836787) q[2];
ry(1.7743606202681992) q[4];
cx q[2],q[4];
ry(0.10301754021771002) q[2];
ry(-3.14037877553839) q[4];
cx q[2],q[4];
ry(0.7322395602510738) q[4];
ry(1.4216733693066566) q[6];
cx q[4],q[6];
ry(-3.132314305837149) q[4];
ry(0.013420555150901445) q[6];
cx q[4],q[6];
ry(-2.5522961116149285) q[6];
ry(0.2636374731695541) q[8];
cx q[6],q[8];
ry(-1.9676208697091218) q[6];
ry(0.14292018022755515) q[8];
cx q[6],q[8];
ry(0.38702322926770627) q[8];
ry(-2.469852957676843) q[10];
cx q[8],q[10];
ry(-0.4948402056034844) q[8];
ry(-0.13783265657805457) q[10];
cx q[8],q[10];
ry(3.0570009616485683) q[1];
ry(0.18825351139165925) q[3];
cx q[1],q[3];
ry(0.19213072735355663) q[1];
ry(-1.2874701816930545) q[3];
cx q[1],q[3];
ry(-3.063057240909853) q[3];
ry(-1.4677531605813754) q[5];
cx q[3],q[5];
ry(-0.00832865837749173) q[3];
ry(-0.003488485564223513) q[5];
cx q[3],q[5];
ry(-0.6376581620854909) q[5];
ry(-3.0893173344441323) q[7];
cx q[5],q[7];
ry(-0.032146447419082506) q[5];
ry(3.0368806985199597) q[7];
cx q[5],q[7];
ry(1.3879381644483404) q[7];
ry(2.7705320146971864) q[9];
cx q[7],q[9];
ry(-0.029446238874228033) q[7];
ry(-3.0956858696887037) q[9];
cx q[7],q[9];
ry(0.3831328522509212) q[9];
ry(0.014003603176049382) q[11];
cx q[9],q[11];
ry(2.295034320138032) q[9];
ry(-1.9228843107393534) q[11];
cx q[9],q[11];
ry(1.3613384243574584) q[0];
ry(1.991921280233135) q[3];
cx q[0],q[3];
ry(-2.83201310721117) q[0];
ry(-1.658944395870064) q[3];
cx q[0],q[3];
ry(-2.120364973401144) q[1];
ry(-0.23503491188651515) q[2];
cx q[1],q[2];
ry(0.4890386081458061) q[1];
ry(2.466958257228172) q[2];
cx q[1],q[2];
ry(1.768144247397824) q[2];
ry(1.1809467231943058) q[5];
cx q[2],q[5];
ry(-0.020335573127899458) q[2];
ry(3.1140499504664243) q[5];
cx q[2],q[5];
ry(0.32720621351670837) q[3];
ry(-0.3307441865332024) q[4];
cx q[3],q[4];
ry(0.08408966492118443) q[3];
ry(-0.02403449510681886) q[4];
cx q[3],q[4];
ry(-1.7229846653883403) q[4];
ry(1.3549013425081557) q[7];
cx q[4],q[7];
ry(3.1073911675127213) q[4];
ry(3.118123225221504) q[7];
cx q[4],q[7];
ry(1.5341187459951975) q[5];
ry(1.3374360209947322) q[6];
cx q[5],q[6];
ry(-2.7409628312513163) q[5];
ry(-0.44650591772649406) q[6];
cx q[5],q[6];
ry(1.6096923766968925) q[6];
ry(0.7907324101549884) q[9];
cx q[6],q[9];
ry(-0.03543896630076596) q[6];
ry(-0.030521086482415236) q[9];
cx q[6],q[9];
ry(-1.1249223585442487) q[7];
ry(1.096979210625766) q[8];
cx q[7],q[8];
ry(2.9849121831384924) q[7];
ry(-2.949739653025641) q[8];
cx q[7],q[8];
ry(3.1163628395908916) q[8];
ry(-0.1435799755821189) q[11];
cx q[8],q[11];
ry(2.6330541909325427) q[8];
ry(0.042967545973182954) q[11];
cx q[8],q[11];
ry(0.9054145517259782) q[9];
ry(-0.13855436112963704) q[10];
cx q[9],q[10];
ry(-2.701373630550399) q[9];
ry(-2.2646893053151187) q[10];
cx q[9],q[10];
ry(0.7709430775876845) q[0];
ry(1.5042314228806237) q[1];
ry(1.670162048899118) q[2];
ry(1.8746160316501168) q[3];
ry(-1.320341154869526) q[4];
ry(-1.6085563348719667) q[5];
ry(1.6345043893315623) q[6];
ry(2.069326652732932) q[7];
ry(-0.11792470598625705) q[8];
ry(1.337844404837237) q[9];
ry(-3.0534061243779638) q[10];
ry(-2.9596745519485825) q[11];