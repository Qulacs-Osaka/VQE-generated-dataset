OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.8927230529099122) q[0];
ry(2.0356750386213758) q[1];
cx q[0],q[1];
ry(-3.1302756764952133) q[0];
ry(0.5196874781140641) q[1];
cx q[0],q[1];
ry(0.8446359485928161) q[2];
ry(0.22488947445337892) q[3];
cx q[2],q[3];
ry(1.2330069927442604) q[2];
ry(-2.0915430763064577) q[3];
cx q[2],q[3];
ry(0.6241415333152478) q[0];
ry(-2.2117671747171435) q[2];
cx q[0],q[2];
ry(1.0649757789193715) q[0];
ry(-1.9610684882402758) q[2];
cx q[0],q[2];
ry(2.477901777483632) q[1];
ry(-2.6956306439314592) q[3];
cx q[1],q[3];
ry(-1.8430069042946664) q[1];
ry(-0.2788949178574134) q[3];
cx q[1],q[3];
ry(3.0521878323159672) q[0];
ry(1.1587046457486707) q[1];
cx q[0],q[1];
ry(0.6770237131212511) q[0];
ry(0.48436950156080183) q[1];
cx q[0],q[1];
ry(2.4287557997958764) q[2];
ry(-1.5025985963851662) q[3];
cx q[2],q[3];
ry(-2.7899157585901126) q[2];
ry(1.3499977465844717) q[3];
cx q[2],q[3];
ry(0.6579163576932868) q[0];
ry(-1.7783985073176956) q[2];
cx q[0],q[2];
ry(-1.5731030473506797) q[0];
ry(2.8524520759621423) q[2];
cx q[0],q[2];
ry(-1.1136992240837316) q[1];
ry(-0.8774208911283701) q[3];
cx q[1],q[3];
ry(1.2147022183738185) q[1];
ry(0.8461538885411338) q[3];
cx q[1],q[3];
ry(1.1510432834781845) q[0];
ry(-0.6760466071867733) q[1];
cx q[0],q[1];
ry(-1.6057132417660562) q[0];
ry(-1.2581760768559878) q[1];
cx q[0],q[1];
ry(1.558502762627904) q[2];
ry(-1.6023952185609085) q[3];
cx q[2],q[3];
ry(0.1356116084588885) q[2];
ry(2.5304704903521946) q[3];
cx q[2],q[3];
ry(-2.21811063892057) q[0];
ry(2.6879158964354595) q[2];
cx q[0],q[2];
ry(-2.838291989021529) q[0];
ry(2.987187514375973) q[2];
cx q[0],q[2];
ry(0.05848635560436133) q[1];
ry(0.6868121047245062) q[3];
cx q[1],q[3];
ry(0.30618185236612394) q[1];
ry(-1.0999877393977666) q[3];
cx q[1],q[3];
ry(2.1430302930332914) q[0];
ry(-0.949734540724244) q[1];
cx q[0],q[1];
ry(-0.6132216371228845) q[0];
ry(0.7120116759548836) q[1];
cx q[0],q[1];
ry(1.3208049839916136) q[2];
ry(0.8462015962079679) q[3];
cx q[2],q[3];
ry(2.7903726331418772) q[2];
ry(2.137493909023102) q[3];
cx q[2],q[3];
ry(0.13199631852014088) q[0];
ry(2.019919122520071) q[2];
cx q[0],q[2];
ry(0.5926037706084082) q[0];
ry(-2.09692522665199) q[2];
cx q[0],q[2];
ry(3.076667029975611) q[1];
ry(0.8074334613545773) q[3];
cx q[1],q[3];
ry(-2.7138971641388614) q[1];
ry(2.805753137362793) q[3];
cx q[1],q[3];
ry(2.302963071377957) q[0];
ry(0.5392676025902504) q[1];
cx q[0],q[1];
ry(0.8375690996086166) q[0];
ry(-1.5790287366857614) q[1];
cx q[0],q[1];
ry(1.494710456513288) q[2];
ry(0.5064357516371134) q[3];
cx q[2],q[3];
ry(2.5277166849017045) q[2];
ry(2.95056219422087) q[3];
cx q[2],q[3];
ry(-1.3124698290430876) q[0];
ry(-1.7759809283204018) q[2];
cx q[0],q[2];
ry(-0.38680968380795216) q[0];
ry(2.214928767585871) q[2];
cx q[0],q[2];
ry(2.678399361168452) q[1];
ry(-2.370994639591632) q[3];
cx q[1],q[3];
ry(-1.797877748563355) q[1];
ry(-0.16594642856835656) q[3];
cx q[1],q[3];
ry(-1.0798868220132656) q[0];
ry(2.123004864628096) q[1];
cx q[0],q[1];
ry(-0.12790649438910145) q[0];
ry(1.9861009311412021) q[1];
cx q[0],q[1];
ry(1.2581135154470018) q[2];
ry(-1.4947541457500293) q[3];
cx q[2],q[3];
ry(2.281611134036552) q[2];
ry(-2.3212411460623157) q[3];
cx q[2],q[3];
ry(-1.4462526827404556) q[0];
ry(2.766423301422792) q[2];
cx q[0],q[2];
ry(-1.2907222878411393) q[0];
ry(1.0913474574595392) q[2];
cx q[0],q[2];
ry(2.6575743208703506) q[1];
ry(-2.022309216350587) q[3];
cx q[1],q[3];
ry(0.3132052891976596) q[1];
ry(-1.8881085136903035) q[3];
cx q[1],q[3];
ry(-1.5179375948528209) q[0];
ry(0.39979614361141635) q[1];
cx q[0],q[1];
ry(2.9357266275067144) q[0];
ry(-0.9338598914777482) q[1];
cx q[0],q[1];
ry(-2.019155747471146) q[2];
ry(-1.4159583804216744) q[3];
cx q[2],q[3];
ry(2.203428725940444) q[2];
ry(-2.7257606376988304) q[3];
cx q[2],q[3];
ry(1.5584571540443415) q[0];
ry(0.637966604326369) q[2];
cx q[0],q[2];
ry(2.5109717450657203) q[0];
ry(-2.0747526555813423) q[2];
cx q[0],q[2];
ry(-1.3122732707917546) q[1];
ry(-0.7826783157188704) q[3];
cx q[1],q[3];
ry(-3.0062708303953647) q[1];
ry(-2.3992998773932146) q[3];
cx q[1],q[3];
ry(-2.8513013545765036) q[0];
ry(-0.2782116485144579) q[1];
cx q[0],q[1];
ry(-0.9241902785392171) q[0];
ry(0.10666829547335865) q[1];
cx q[0],q[1];
ry(-2.4810943702279906) q[2];
ry(-0.10870305222643203) q[3];
cx q[2],q[3];
ry(-2.620098448384831) q[2];
ry(-1.33372802673805) q[3];
cx q[2],q[3];
ry(1.3756066604176622) q[0];
ry(-1.756479337707291) q[2];
cx q[0],q[2];
ry(2.2060092317147797) q[0];
ry(-2.210185612018903) q[2];
cx q[0],q[2];
ry(2.926392886904432) q[1];
ry(2.9485207321283635) q[3];
cx q[1],q[3];
ry(2.910687099477886) q[1];
ry(-2.978170745996039) q[3];
cx q[1],q[3];
ry(0.882306687757394) q[0];
ry(-1.9907887599582863) q[1];
cx q[0],q[1];
ry(3.0235173839701366) q[0];
ry(-2.0269973001927224) q[1];
cx q[0],q[1];
ry(2.098324858754795) q[2];
ry(2.892260089742345) q[3];
cx q[2],q[3];
ry(1.207371910696387) q[2];
ry(2.0022086358669666) q[3];
cx q[2],q[3];
ry(-1.4191355819291847) q[0];
ry(1.749167078513361) q[2];
cx q[0],q[2];
ry(1.3357141887043786) q[0];
ry(1.5076234976464604) q[2];
cx q[0],q[2];
ry(-2.085723439472794) q[1];
ry(-0.9453830003741991) q[3];
cx q[1],q[3];
ry(2.3496092507819197) q[1];
ry(-0.795972506936024) q[3];
cx q[1],q[3];
ry(0.9115309931398127) q[0];
ry(-2.803354360427936) q[1];
cx q[0],q[1];
ry(2.7792961831539866) q[0];
ry(-2.7245416604460564) q[1];
cx q[0],q[1];
ry(1.112403768800065) q[2];
ry(2.295562895125086) q[3];
cx q[2],q[3];
ry(2.5586662572487966) q[2];
ry(-1.6799457167608407) q[3];
cx q[2],q[3];
ry(-2.741285229173221) q[0];
ry(-2.702182467232757) q[2];
cx q[0],q[2];
ry(2.25305106060169) q[0];
ry(-1.5292071839205759) q[2];
cx q[0],q[2];
ry(-0.2256402779582993) q[1];
ry(0.5482318461430884) q[3];
cx q[1],q[3];
ry(-2.993233316237476) q[1];
ry(1.8944344837175506) q[3];
cx q[1],q[3];
ry(0.728177968643231) q[0];
ry(2.280852043685406) q[1];
cx q[0],q[1];
ry(2.6951733296210985) q[0];
ry(0.6088359142914879) q[1];
cx q[0],q[1];
ry(-3.009354407723182) q[2];
ry(-1.888610588768674) q[3];
cx q[2],q[3];
ry(-0.44148195252335487) q[2];
ry(0.5505382456076378) q[3];
cx q[2],q[3];
ry(0.511383454945021) q[0];
ry(-1.190848611373821) q[2];
cx q[0],q[2];
ry(-2.9162654286881677) q[0];
ry(-1.300298973249438) q[2];
cx q[0],q[2];
ry(-3.0361475783961067) q[1];
ry(-1.7620071193693698) q[3];
cx q[1],q[3];
ry(-2.8581880143996825) q[1];
ry(-2.2959559658496556) q[3];
cx q[1],q[3];
ry(-0.8705497663262383) q[0];
ry(2.0691597225568668) q[1];
cx q[0],q[1];
ry(0.9489994072193148) q[0];
ry(0.48842894714028806) q[1];
cx q[0],q[1];
ry(-2.460199602554563) q[2];
ry(1.4415156146372672) q[3];
cx q[2],q[3];
ry(-3.1267017640848613) q[2];
ry(3.0857842305696614) q[3];
cx q[2],q[3];
ry(-1.9100008659868513) q[0];
ry(2.9067130761791082) q[2];
cx q[0],q[2];
ry(-1.167797022406253) q[0];
ry(-1.3797370017906019) q[2];
cx q[0],q[2];
ry(-2.3493435092422166) q[1];
ry(-0.5734063600003312) q[3];
cx q[1],q[3];
ry(2.218577174804576) q[1];
ry(-0.43694039331189405) q[3];
cx q[1],q[3];
ry(0.835823198199666) q[0];
ry(-0.4750374211344627) q[1];
cx q[0],q[1];
ry(0.053015564343467825) q[0];
ry(-0.6938384468644667) q[1];
cx q[0],q[1];
ry(-2.7214523778929567) q[2];
ry(-2.932138772118) q[3];
cx q[2],q[3];
ry(2.9428051399067763) q[2];
ry(2.748706787231427) q[3];
cx q[2],q[3];
ry(2.592426058235412) q[0];
ry(2.4675462375451707) q[2];
cx q[0],q[2];
ry(0.0625218998204189) q[0];
ry(0.6049001010587949) q[2];
cx q[0],q[2];
ry(0.15242898240530064) q[1];
ry(-0.7863293335892392) q[3];
cx q[1],q[3];
ry(0.21642524495859128) q[1];
ry(2.2948538893863732) q[3];
cx q[1],q[3];
ry(-0.734854854623303) q[0];
ry(-2.077226379464065) q[1];
cx q[0],q[1];
ry(-0.06394975895143444) q[0];
ry(-0.3941696996792776) q[1];
cx q[0],q[1];
ry(0.3230754434771317) q[2];
ry(-0.6707583939040127) q[3];
cx q[2],q[3];
ry(1.8779996384201043) q[2];
ry(-1.4928641431634755) q[3];
cx q[2],q[3];
ry(3.020583829122245) q[0];
ry(-1.1887379366566593) q[2];
cx q[0],q[2];
ry(2.5634983007712684) q[0];
ry(2.1289104630847775) q[2];
cx q[0],q[2];
ry(2.7595326404788953) q[1];
ry(-3.116344522176115) q[3];
cx q[1],q[3];
ry(-2.4666514395685692) q[1];
ry(0.36422513304571336) q[3];
cx q[1],q[3];
ry(2.479016927740973) q[0];
ry(-1.8631204082933444) q[1];
cx q[0],q[1];
ry(-2.2938792336290676) q[0];
ry(-0.5372788270352381) q[1];
cx q[0],q[1];
ry(-2.1360761132923924) q[2];
ry(2.596329989538719) q[3];
cx q[2],q[3];
ry(0.6035942342934955) q[2];
ry(1.6799683971620514) q[3];
cx q[2],q[3];
ry(2.2154283113063684) q[0];
ry(2.493052632324689) q[2];
cx q[0],q[2];
ry(1.9831414250941373) q[0];
ry(-2.5915306966257243) q[2];
cx q[0],q[2];
ry(2.823157762669719) q[1];
ry(-1.3433660183692369) q[3];
cx q[1],q[3];
ry(-0.6500267752001507) q[1];
ry(2.737695675125187) q[3];
cx q[1],q[3];
ry(-2.8593418245405062) q[0];
ry(2.7848920570568456) q[1];
cx q[0],q[1];
ry(-1.5970058574062922) q[0];
ry(2.1440550713318434) q[1];
cx q[0],q[1];
ry(-2.5726599422966325) q[2];
ry(2.3513810572602085) q[3];
cx q[2],q[3];
ry(0.891692566295565) q[2];
ry(-1.8210519208006548) q[3];
cx q[2],q[3];
ry(-1.7005167122331581) q[0];
ry(1.2898228474027222) q[2];
cx q[0],q[2];
ry(-2.1993300918364147) q[0];
ry(2.8153004881383903) q[2];
cx q[0],q[2];
ry(-0.1324695947736251) q[1];
ry(0.938215335155769) q[3];
cx q[1],q[3];
ry(-2.4873290038302245) q[1];
ry(0.8821606944321353) q[3];
cx q[1],q[3];
ry(0.9522182116766462) q[0];
ry(-1.2585114527544006) q[1];
cx q[0],q[1];
ry(-0.12557465743693272) q[0];
ry(0.3167839126051995) q[1];
cx q[0],q[1];
ry(-2.716154130514415) q[2];
ry(0.2835369395897901) q[3];
cx q[2],q[3];
ry(0.9513268588695766) q[2];
ry(1.3983969291066818) q[3];
cx q[2],q[3];
ry(-0.0850900051604922) q[0];
ry(-2.302265509429116) q[2];
cx q[0],q[2];
ry(0.28099985246256554) q[0];
ry(2.357800077953852) q[2];
cx q[0],q[2];
ry(1.036281034662287) q[1];
ry(0.8059423411186667) q[3];
cx q[1],q[3];
ry(1.983034801134874) q[1];
ry(-2.986012165225864) q[3];
cx q[1],q[3];
ry(-0.44824022506090166) q[0];
ry(-3.0207942972529205) q[1];
cx q[0],q[1];
ry(1.982804035800061) q[0];
ry(2.068944373661119) q[1];
cx q[0],q[1];
ry(2.6268595267035457) q[2];
ry(2.5539080907331835) q[3];
cx q[2],q[3];
ry(1.2638260871970322) q[2];
ry(-2.4907643891362237) q[3];
cx q[2],q[3];
ry(0.5578595313268793) q[0];
ry(-2.286674547367752) q[2];
cx q[0],q[2];
ry(0.8626325966679528) q[0];
ry(1.3158576435032128) q[2];
cx q[0],q[2];
ry(-1.0966957047151358) q[1];
ry(2.1694753128795585) q[3];
cx q[1],q[3];
ry(-1.0973506815397824) q[1];
ry(-2.0655454860504614) q[3];
cx q[1],q[3];
ry(0.9431864301333449) q[0];
ry(-0.29574280382773316) q[1];
cx q[0],q[1];
ry(-0.47366844985402246) q[0];
ry(2.490261995393341) q[1];
cx q[0],q[1];
ry(1.2855264324561606) q[2];
ry(0.8881322833014207) q[3];
cx q[2],q[3];
ry(2.8596673078478987) q[2];
ry(0.1440207039913881) q[3];
cx q[2],q[3];
ry(-1.9458394972116515) q[0];
ry(0.9768033265868344) q[2];
cx q[0],q[2];
ry(-0.7372838618617329) q[0];
ry(1.7109476322708765) q[2];
cx q[0],q[2];
ry(-2.6513239962676587) q[1];
ry(-2.1664191220842977) q[3];
cx q[1],q[3];
ry(1.8438755832898384) q[1];
ry(-1.5876451146225194) q[3];
cx q[1],q[3];
ry(2.2922183412614183) q[0];
ry(-2.0912317339453916) q[1];
cx q[0],q[1];
ry(1.0986172316631888) q[0];
ry(-2.6784934889648246) q[1];
cx q[0],q[1];
ry(-2.151195282653697) q[2];
ry(1.3262350008526305) q[3];
cx q[2],q[3];
ry(-2.7337597410396155) q[2];
ry(-2.0071504695887996) q[3];
cx q[2],q[3];
ry(0.7649538264746438) q[0];
ry(-2.9647752479316964) q[2];
cx q[0],q[2];
ry(-0.9460912010526973) q[0];
ry(-0.6407643753964667) q[2];
cx q[0],q[2];
ry(-2.6395088154581843) q[1];
ry(1.1312489399713035) q[3];
cx q[1],q[3];
ry(2.6322233567948583) q[1];
ry(-0.3883957687745064) q[3];
cx q[1],q[3];
ry(2.287905156924439) q[0];
ry(3.0973899769855966) q[1];
cx q[0],q[1];
ry(-1.7788137101266255) q[0];
ry(-1.325232078275861) q[1];
cx q[0],q[1];
ry(-2.24425110801587) q[2];
ry(-1.0533244875274985) q[3];
cx q[2],q[3];
ry(-2.979071692354499) q[2];
ry(-0.6233552885425193) q[3];
cx q[2],q[3];
ry(3.0961739683411817) q[0];
ry(-2.6905206109085213) q[2];
cx q[0],q[2];
ry(-2.4329482060950314) q[0];
ry(2.679588845247574) q[2];
cx q[0],q[2];
ry(1.1290666354467325) q[1];
ry(-2.6246710693339597) q[3];
cx q[1],q[3];
ry(1.1912829788657096) q[1];
ry(0.12083479117674756) q[3];
cx q[1],q[3];
ry(-2.8427442632238114) q[0];
ry(1.6689557419748073) q[1];
cx q[0],q[1];
ry(0.5332990225668396) q[0];
ry(-0.05189543487437299) q[1];
cx q[0],q[1];
ry(-2.2634023585797) q[2];
ry(-1.5995399537059802) q[3];
cx q[2],q[3];
ry(-2.2439174521635965) q[2];
ry(-1.6779394518571107) q[3];
cx q[2],q[3];
ry(1.8531754546738266) q[0];
ry(-2.8747095343826294) q[2];
cx q[0],q[2];
ry(-2.776443182373928) q[0];
ry(0.679530134164013) q[2];
cx q[0],q[2];
ry(2.8914042531035666) q[1];
ry(-0.7304074741941439) q[3];
cx q[1],q[3];
ry(1.5645038428477367) q[1];
ry(-2.774557328998871) q[3];
cx q[1],q[3];
ry(-2.7817036788054073) q[0];
ry(-1.7159275510458705) q[1];
cx q[0],q[1];
ry(1.0336643742030598) q[0];
ry(0.5483700989310965) q[1];
cx q[0],q[1];
ry(0.8393746581848409) q[2];
ry(-1.3941803979822893) q[3];
cx q[2],q[3];
ry(0.5949436462434532) q[2];
ry(1.457979815977759) q[3];
cx q[2],q[3];
ry(-3.1044673982427353) q[0];
ry(2.990242116832436) q[2];
cx q[0],q[2];
ry(-2.148366194256803) q[0];
ry(-1.8208399648107638) q[2];
cx q[0],q[2];
ry(-2.2145654476910046) q[1];
ry(2.8396953153338393) q[3];
cx q[1],q[3];
ry(0.8226523015399684) q[1];
ry(1.5347887734082046) q[3];
cx q[1],q[3];
ry(-1.9958627434283214) q[0];
ry(-0.3078920426986542) q[1];
cx q[0],q[1];
ry(-1.9423115694609525) q[0];
ry(-2.5751687722977845) q[1];
cx q[0],q[1];
ry(-2.6052589011321476) q[2];
ry(0.46502182214503573) q[3];
cx q[2],q[3];
ry(0.6194319083585871) q[2];
ry(1.4205360902771478) q[3];
cx q[2],q[3];
ry(-1.7986254033059472) q[0];
ry(-0.6724630121467436) q[2];
cx q[0],q[2];
ry(0.7644329603446742) q[0];
ry(0.6502876302713787) q[2];
cx q[0],q[2];
ry(-0.8367506653480351) q[1];
ry(-3.07982482613816) q[3];
cx q[1],q[3];
ry(-2.364666523159769) q[1];
ry(0.8466551855857258) q[3];
cx q[1],q[3];
ry(-1.8211896441598536) q[0];
ry(-1.426267225789645) q[1];
cx q[0],q[1];
ry(1.0408980343143153) q[0];
ry(-2.815484707503428) q[1];
cx q[0],q[1];
ry(2.8067457444177033) q[2];
ry(2.323466368713939) q[3];
cx q[2],q[3];
ry(-2.2313129190213203) q[2];
ry(-2.7470572575059853) q[3];
cx q[2],q[3];
ry(2.0499719419934963) q[0];
ry(2.997721538826616) q[2];
cx q[0],q[2];
ry(-2.2482113233296452) q[0];
ry(1.9688624694264982) q[2];
cx q[0],q[2];
ry(0.31326634229417927) q[1];
ry(-2.3651048957467307) q[3];
cx q[1],q[3];
ry(-2.207536681190889) q[1];
ry(-2.7395839761584737) q[3];
cx q[1],q[3];
ry(-1.1837123106773397) q[0];
ry(-2.8125104541668606) q[1];
cx q[0],q[1];
ry(2.4847867372895323) q[0];
ry(3.0763634939448217) q[1];
cx q[0],q[1];
ry(0.11508447779358733) q[2];
ry(-0.2742740047651221) q[3];
cx q[2],q[3];
ry(1.5899212113560859) q[2];
ry(-1.8001806803687435) q[3];
cx q[2],q[3];
ry(1.0157668170692082) q[0];
ry(2.713064711421209) q[2];
cx q[0],q[2];
ry(2.305803702208244) q[0];
ry(1.409288880994938) q[2];
cx q[0],q[2];
ry(1.3017866368166742) q[1];
ry(-0.29081316174166033) q[3];
cx q[1],q[3];
ry(0.8496167259957303) q[1];
ry(0.1097814417255556) q[3];
cx q[1],q[3];
ry(0.42405352833718984) q[0];
ry(0.33064735277033375) q[1];
cx q[0],q[1];
ry(1.362981247192136) q[0];
ry(3.0826968966890957) q[1];
cx q[0],q[1];
ry(2.4920780322799794) q[2];
ry(0.09975484308062475) q[3];
cx q[2],q[3];
ry(-2.0407211228211275) q[2];
ry(1.9796424157513322) q[3];
cx q[2],q[3];
ry(0.0821318782131577) q[0];
ry(-2.0011396175479903) q[2];
cx q[0],q[2];
ry(-2.628838393917416) q[0];
ry(-0.06882491253128455) q[2];
cx q[0],q[2];
ry(1.0648718367823484) q[1];
ry(-2.9158653134461154) q[3];
cx q[1],q[3];
ry(0.17229235038127658) q[1];
ry(-1.2218405772510432) q[3];
cx q[1],q[3];
ry(1.3860165585968112) q[0];
ry(-1.5395535385340982) q[1];
cx q[0],q[1];
ry(-2.6680715195552107) q[0];
ry(1.354222306522419) q[1];
cx q[0],q[1];
ry(2.7379034805867093) q[2];
ry(-0.07049549655943733) q[3];
cx q[2],q[3];
ry(-0.2808252521668768) q[2];
ry(-0.1189570115930243) q[3];
cx q[2],q[3];
ry(-1.41255970215295) q[0];
ry(-2.173013610010611) q[2];
cx q[0],q[2];
ry(-1.2659930945849696) q[0];
ry(-2.0847876486729007) q[2];
cx q[0],q[2];
ry(-2.036091393151172) q[1];
ry(1.5598114859030894) q[3];
cx q[1],q[3];
ry(2.359527879346841) q[1];
ry(0.2865125665676777) q[3];
cx q[1],q[3];
ry(0.23930279081514882) q[0];
ry(1.5914035579090848) q[1];
cx q[0],q[1];
ry(-1.856835039747393) q[0];
ry(1.6252529314473074) q[1];
cx q[0],q[1];
ry(-1.0688109063728586) q[2];
ry(1.4559463090423934) q[3];
cx q[2],q[3];
ry(-1.3391013788872492) q[2];
ry(1.369197524254807) q[3];
cx q[2],q[3];
ry(2.050071648203212) q[0];
ry(-3.031309719655841) q[2];
cx q[0],q[2];
ry(-2.8305556407839467) q[0];
ry(1.9276357884025197) q[2];
cx q[0],q[2];
ry(0.07248500546229253) q[1];
ry(-1.943534254548108) q[3];
cx q[1],q[3];
ry(1.629720941130417) q[1];
ry(0.5201697832352575) q[3];
cx q[1],q[3];
ry(-2.7651079962239815) q[0];
ry(2.6355682749905736) q[1];
cx q[0],q[1];
ry(2.532493494034872) q[0];
ry(1.288126773805882) q[1];
cx q[0],q[1];
ry(-0.3498599730339254) q[2];
ry(-1.1527754751278199) q[3];
cx q[2],q[3];
ry(0.41165331744477485) q[2];
ry(1.8615372363446956) q[3];
cx q[2],q[3];
ry(0.6141771812609989) q[0];
ry(2.7911190830908126) q[2];
cx q[0],q[2];
ry(1.6058731010286833) q[0];
ry(1.989146710001446) q[2];
cx q[0],q[2];
ry(0.3466398388763112) q[1];
ry(-0.6962392950698569) q[3];
cx q[1],q[3];
ry(-2.1249803855742675) q[1];
ry(2.851027689823049) q[3];
cx q[1],q[3];
ry(-1.6825609581643208) q[0];
ry(-1.414290591617064) q[1];
cx q[0],q[1];
ry(-0.12260438053609313) q[0];
ry(1.622678566656957) q[1];
cx q[0],q[1];
ry(-0.32397576705757225) q[2];
ry(-2.51184707781602) q[3];
cx q[2],q[3];
ry(0.9769340631358486) q[2];
ry(2.226256360871697) q[3];
cx q[2],q[3];
ry(-1.8882137679663442) q[0];
ry(0.04105808727800486) q[2];
cx q[0],q[2];
ry(-1.0908460976757892) q[0];
ry(-2.758941623809486) q[2];
cx q[0],q[2];
ry(0.8880246081149565) q[1];
ry(2.9885790791958975) q[3];
cx q[1],q[3];
ry(-0.2748305441614214) q[1];
ry(-2.354270481948089) q[3];
cx q[1],q[3];
ry(2.221832125017699) q[0];
ry(-2.3502430462954953) q[1];
cx q[0],q[1];
ry(-1.366172344913032) q[0];
ry(-1.8168085006006898) q[1];
cx q[0],q[1];
ry(0.9394846365118236) q[2];
ry(-2.404281646287008) q[3];
cx q[2],q[3];
ry(-1.7308784740926892) q[2];
ry(0.33906614293910026) q[3];
cx q[2],q[3];
ry(0.9696891046664043) q[0];
ry(-2.970680164217163) q[2];
cx q[0],q[2];
ry(1.2982304111050755) q[0];
ry(0.8960456911889666) q[2];
cx q[0],q[2];
ry(-2.547809246500189) q[1];
ry(-1.222851051472899) q[3];
cx q[1],q[3];
ry(0.4374354169537362) q[1];
ry(-0.09150472816988044) q[3];
cx q[1],q[3];
ry(1.6016748956546676) q[0];
ry(-2.518531507566735) q[1];
ry(-1.492330413456931) q[2];
ry(-0.523313075864593) q[3];