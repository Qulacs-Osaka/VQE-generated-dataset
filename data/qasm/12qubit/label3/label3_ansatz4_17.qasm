OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.6481124689326387) q[0];
rz(3.07586126270043) q[0];
ry(-0.9326011480320823) q[1];
rz(2.8906047162005266) q[1];
ry(-1.633552261705062) q[2];
rz(2.676544464076719) q[2];
ry(1.240833207226863) q[3];
rz(-0.03860407790803325) q[3];
ry(-3.139796697788603) q[4];
rz(-0.32136208001489214) q[4];
ry(0.0012602167476698511) q[5];
rz(1.6993606359464764) q[5];
ry(-1.5652622579567146) q[6];
rz(-1.7217082947972164) q[6];
ry(-1.5558876654764602) q[7];
rz(1.5477530793090324) q[7];
ry(2.9069800331287334) q[8];
rz(0.7577365178410799) q[8];
ry(-0.15993315734187794) q[9];
rz(-1.303134284793857) q[9];
ry(-1.0626613608083915) q[10];
rz(-2.1248586852144875) q[10];
ry(1.3900574752361152) q[11];
rz(-1.23669667339097) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.4937489983616472) q[0];
rz(0.4672021746691327) q[0];
ry(1.926459909550465) q[1];
rz(2.048163454421119) q[1];
ry(0.8555585930666902) q[2];
rz(-1.2981733485034497) q[2];
ry(0.39588617995387815) q[3];
rz(-1.756319168259047) q[3];
ry(0.4142885099296434) q[4];
rz(1.7475198046725904) q[4];
ry(-0.6159630263554771) q[5];
rz(-0.010324219233580954) q[5];
ry(-1.5405994977204527) q[6];
rz(-0.11871416414432012) q[6];
ry(-1.7221672164536637) q[7];
rz(-1.1550863928239616) q[7];
ry(1.4414019600593289) q[8];
rz(-1.0902903205907135) q[8];
ry(-1.0592873505333875) q[9];
rz(-0.5700701418803266) q[9];
ry(0.3303663422548106) q[10];
rz(-2.5891986884092444) q[10];
ry(0.4374663360510409) q[11];
rz(2.906204285328518) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.1375337867907804) q[0];
rz(0.9548408232609251) q[0];
ry(1.364862831063851) q[1];
rz(2.0636097157152475) q[1];
ry(-0.7288560903711705) q[2];
rz(1.9948585939886325) q[2];
ry(0.5593511167481738) q[3];
rz(-1.5616753020804026) q[3];
ry(3.139972198881796) q[4];
rz(-0.4948605654273246) q[4];
ry(-0.10922926977621082) q[5];
rz(-1.4568492125572303) q[5];
ry(3.1322690888966247) q[6];
rz(-2.3276747820251935) q[6];
ry(0.02017147556750576) q[7];
rz(2.7998110663242453) q[7];
ry(1.9266423237555257) q[8];
rz(-1.9199534587452831) q[8];
ry(0.9253688538751705) q[9];
rz(1.4902661420354046) q[9];
ry(1.628018516279526) q[10];
rz(-1.0641979820749494) q[10];
ry(-1.5604758545475899) q[11];
rz(1.8152617886475628) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.976472198666383) q[0];
rz(-2.185385394382844) q[0];
ry(-0.6680041300629487) q[1];
rz(-2.574896968011668) q[1];
ry(0.11085859519574007) q[2];
rz(-2.152836313929898) q[2];
ry(-1.8831592102371337) q[3];
rz(-1.6255783109712194) q[3];
ry(2.3356784756935958) q[4];
rz(1.7331213107189587) q[4];
ry(0.4495594547429273) q[5];
rz(-0.8726085020481095) q[5];
ry(-0.03713428687415689) q[6];
rz(-2.129492824593159) q[6];
ry(3.0798796050047583) q[7];
rz(2.1207301849576283) q[7];
ry(-2.991211913467887) q[8];
rz(-2.369239597779956) q[8];
ry(0.2401963902542894) q[9];
rz(-2.17230672058521) q[9];
ry(-2.520847891728343) q[10];
rz(2.820948438690729) q[10];
ry(1.5614739275397136) q[11];
rz(1.3586078944936892) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.0557843294311655) q[0];
rz(1.698445110991276) q[0];
ry(0.15243216228823483) q[1];
rz(1.9895941309675138) q[1];
ry(-1.9951211247365594) q[2];
rz(1.5733329382519567) q[2];
ry(-2.028673978603754) q[3];
rz(-1.8686535678401373) q[3];
ry(-1.5929786117638836) q[4];
rz(-2.3505859969723293) q[4];
ry(1.4978004097792885) q[5];
rz(-1.72694426060186) q[5];
ry(-3.126352569025997) q[6];
rz(1.5205586347197038) q[6];
ry(-0.031647514775083437) q[7];
rz(0.3414588764849199) q[7];
ry(-2.166754280596244) q[8];
rz(2.8656743887521023) q[8];
ry(-0.49321265198183273) q[9];
rz(2.9374318230557988) q[9];
ry(-1.0219163162676725) q[10];
rz(2.1640177457438794) q[10];
ry(-2.672113953962697) q[11];
rz(2.7663346854893898) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.4409905044431914) q[0];
rz(1.7327488831027598) q[0];
ry(-1.8099067720229698) q[1];
rz(-0.678403234942083) q[1];
ry(2.0606459251521665) q[2];
rz(2.303832030203595) q[2];
ry(1.1176694801456257) q[3];
rz(2.2459543686408012) q[3];
ry(-3.0731741095750595) q[4];
rz(-1.8429675840301591) q[4];
ry(-0.8960382257284197) q[5];
rz(-1.8977384275658833) q[5];
ry(-3.138545198493406) q[6];
rz(2.4776074342402006) q[6];
ry(-0.0012478352324958055) q[7];
rz(-2.594370360056705) q[7];
ry(-1.2537512857958062) q[8];
rz(-1.7339097250149755) q[8];
ry(-1.8231045455108685) q[9];
rz(0.6054843149537011) q[9];
ry(1.2295478400869004) q[10];
rz(1.206875122065391) q[10];
ry(1.3880182007133541) q[11];
rz(-1.2915254949523403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.08699398751400436) q[0];
rz(-2.9345171662803686) q[0];
ry(-2.129869731910194) q[1];
rz(1.9960463089646483) q[1];
ry(-1.4672070720267967) q[2];
rz(0.6300206845724112) q[2];
ry(1.5727539343796655) q[3];
rz(2.1507466837065046) q[3];
ry(-0.4503488037183727) q[4];
rz(2.7151395078222103) q[4];
ry(-0.8333361948937572) q[5];
rz(-1.9784178437552287) q[5];
ry(1.559575385628352) q[6];
rz(-0.909162649333568) q[6];
ry(1.5797964412571632) q[7];
rz(2.4778630623029954) q[7];
ry(-0.7997538199400545) q[8];
rz(0.6403808341378642) q[8];
ry(2.8445182333828196) q[9];
rz(-1.251461046062932) q[9];
ry(1.4898397065843918) q[10];
rz(-0.6012154344640415) q[10];
ry(-1.3863691694569384) q[11];
rz(-2.403394302080797) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.2139156644158149) q[0];
rz(0.4126075943179851) q[0];
ry(2.890577183014976) q[1];
rz(0.14351206717721732) q[1];
ry(-0.3597482087030777) q[2];
rz(1.8414342435067468) q[2];
ry(0.17691418846524856) q[3];
rz(0.06362437331229726) q[3];
ry(-0.1383643278484774) q[4];
rz(3.0919110569986277) q[4];
ry(3.090417768953626) q[5];
rz(-2.4895491251635704) q[5];
ry(-0.0063303212180674095) q[6];
rz(-2.2359481999815474) q[6];
ry(0.007700393403094808) q[7];
rz(0.6668003138984471) q[7];
ry(-2.5879367724622835) q[8];
rz(1.2100905207364665) q[8];
ry(-1.7027214441757224) q[9];
rz(-0.8729779446199588) q[9];
ry(-2.6912027591410284) q[10];
rz(-0.06028528892125915) q[10];
ry(-0.34478435792784984) q[11];
rz(-0.22324716672799752) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.5227777207322296) q[0];
rz(2.455050337289652) q[0];
ry(1.2579181808457784) q[1];
rz(-0.9249203777209294) q[1];
ry(0.8365259599562962) q[2];
rz(0.6318524514358925) q[2];
ry(2.421923422432092) q[3];
rz(1.9671220471906343) q[3];
ry(-1.9131915557682286) q[4];
rz(0.6067865504629816) q[4];
ry(-0.7384273638990848) q[5];
rz(-2.546786784890137) q[5];
ry(-1.5863502620675503) q[6];
rz(-0.11132409365112793) q[6];
ry(-1.5534675252181565) q[7];
rz(1.4173082649094493) q[7];
ry(-1.66506435416659) q[8];
rz(2.6374678777924965) q[8];
ry(-2.6448615968484965) q[9];
rz(0.4272370082508061) q[9];
ry(-0.4294441826942828) q[10];
rz(1.3944160654717264) q[10];
ry(-1.2777851353358258) q[11];
rz(2.685256044919709) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.9450640887898034) q[0];
rz(2.5194071378490754) q[0];
ry(-0.9668406513326165) q[1];
rz(-0.08546147723170172) q[1];
ry(1.8904213750754604) q[2];
rz(1.0816286350186608) q[2];
ry(1.5999139441291677) q[3];
rz(1.8823452725700003) q[3];
ry(-2.2186309001848397) q[4];
rz(-0.5956733219027025) q[4];
ry(2.392522020197725) q[5];
rz(1.7135773314633802) q[5];
ry(1.6644104222602036) q[6];
rz(1.5515991710658539) q[6];
ry(3.1390971489181623) q[7];
rz(-3.0396390136582565) q[7];
ry(-1.493135620505794) q[8];
rz(0.3340008105075576) q[8];
ry(-0.24845827451712138) q[9];
rz(-2.706133759874261) q[9];
ry(-2.9652739342400154) q[10];
rz(0.28444495330579184) q[10];
ry(1.493953041479343) q[11];
rz(0.6743905870410014) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.3954001525047408) q[0];
rz(-0.6588997506732248) q[0];
ry(2.5030847906797633) q[1];
rz(0.37759741598868146) q[1];
ry(-3.075066512547295) q[2];
rz(1.2404527722192196) q[2];
ry(-0.38760833297695074) q[3];
rz(-0.06507186979084582) q[3];
ry(0.02212440063604959) q[4];
rz(-2.6945052177361966) q[4];
ry(2.7386538149308555) q[5];
rz(-2.5479379775086475) q[5];
ry(-1.2634761660224108) q[6];
rz(-2.9995619761446792) q[6];
ry(2.7966424341833362) q[7];
rz(-1.7654963531474834) q[7];
ry(-1.5784911278422618) q[8];
rz(-1.0641397154031118) q[8];
ry(-1.7257304863950633) q[9];
rz(-1.8749670714445346) q[9];
ry(-1.5472841799838797) q[10];
rz(-2.0974913868529868) q[10];
ry(2.6637739735009314) q[11];
rz(-0.15916939269312813) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.4010577913122098) q[0];
rz(-1.2827333110692862) q[0];
ry(-0.8303997068258537) q[1];
rz(2.821749131968435) q[1];
ry(0.07580213671307288) q[2];
rz(0.2372545546804049) q[2];
ry(2.8821793368921282) q[3];
rz(-2.9214612271741154) q[3];
ry(-3.14114550909325) q[4];
rz(-1.783623396094213) q[4];
ry(-3.141518442087254) q[5];
rz(1.0165746270214546) q[5];
ry(-0.00503366581629372) q[6];
rz(-1.6039842081885871) q[6];
ry(-3.141011135786234) q[7];
rz(2.447312559537264) q[7];
ry(0.2111032418625642) q[8];
rz(2.6031826869523114) q[8];
ry(3.139976031790209) q[9];
rz(-1.7378161085415211) q[9];
ry(3.088754052625131) q[10];
rz(0.5530826289850826) q[10];
ry(2.0871806852870893) q[11];
rz(-1.542848844227971) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.20416365257134217) q[0];
rz(-0.297385310149739) q[0];
ry(1.4273171223912988) q[1];
rz(-1.895514417084451) q[1];
ry(-1.5593612154938927) q[2];
rz(-2.1637219651131483) q[2];
ry(1.7042227175343978) q[3];
rz(-1.4551420987062018) q[3];
ry(1.4649391460978407) q[4];
rz(-2.342218641472791) q[4];
ry(-1.2314252420459335) q[5];
rz(0.8133229392753035) q[5];
ry(-1.5399857741096952) q[6];
rz(1.9882515040642164) q[6];
ry(-2.2099510132434363) q[7];
rz(2.8576449653515157) q[7];
ry(-0.8291081100423394) q[8];
rz(2.982603340889944) q[8];
ry(-0.08269648020235014) q[9];
rz(-0.15129221589911346) q[9];
ry(-1.58480866556026) q[10];
rz(0.2238444577800811) q[10];
ry(-1.5687985101055828) q[11];
rz(-3.030237167772528) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8725303708825773) q[0];
rz(2.6435093677727313) q[0];
ry(-2.484957435001749) q[1];
rz(2.955456490693859) q[1];
ry(-0.028104920384445897) q[2];
rz(2.0668804125980538) q[2];
ry(2.9643192025904623) q[3];
rz(-1.527924126269279) q[3];
ry(0.00019683279353055745) q[4];
rz(2.4464025273928347) q[4];
ry(0.0022406002194531862) q[5];
rz(-2.5346847859674355) q[5];
ry(-0.000399268618544113) q[6];
rz(-0.20786791444091793) q[6];
ry(-3.1410627096590047) q[7];
rz(2.1419284080474648) q[7];
ry(-3.137220157052319) q[8];
rz(1.2331394092107946) q[8];
ry(0.4453794769947957) q[9];
rz(0.015959787266344172) q[9];
ry(1.5767315569398939) q[10];
rz(2.1018238922757595) q[10];
ry(-2.4617384612176116) q[11];
rz(-0.8547408024835362) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.873586440621017) q[0];
rz(-1.8614700636882648) q[0];
ry(-2.7305744974757906) q[1];
rz(-2.277461538021506) q[1];
ry(1.5409402322727357) q[2];
rz(0.5757646934664793) q[2];
ry(1.7322506098268917) q[3];
rz(0.9841155451415169) q[3];
ry(-1.4429751146895944) q[4];
rz(2.7110546937900737) q[4];
ry(-2.1613536627835117) q[5];
rz(2.182902936033317) q[5];
ry(2.9940105532912202) q[6];
rz(-1.787919614494447) q[6];
ry(2.0012232781684567) q[7];
rz(-2.2965640293535228) q[7];
ry(1.572128496899989) q[8];
rz(2.234656548804996) q[8];
ry(-1.4798198493743449) q[9];
rz(-3.0318551123242172) q[9];
ry(-3.105801192983472) q[10];
rz(-1.105956466254455) q[10];
ry(0.004890523493286557) q[11];
rz(1.0384859561801765) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.538109088704692) q[0];
rz(-1.269309360061313) q[0];
ry(-2.947384021181961) q[1];
rz(-1.455991562208732) q[1];
ry(3.0702318647504985) q[2];
rz(2.3893443209457867) q[2];
ry(-1.5681933909359582) q[3];
rz(-0.6327752023251945) q[3];
ry(-3.1343447937657363) q[4];
rz(1.116302231646201) q[4];
ry(3.1390910735362145) q[5];
rz(2.1655995049189394) q[5];
ry(-0.006640430544674558) q[6];
rz(1.6920569939189598) q[6];
ry(1.5722454586118415) q[7];
rz(1.571331247965793) q[7];
ry(0.007773705079705984) q[8];
rz(2.4943644003498298) q[8];
ry(-3.1348786432019513) q[9];
rz(0.11267820733433165) q[9];
ry(0.7738086512832449) q[10];
rz(1.548506361789032) q[10];
ry(-1.5499368126538675) q[11];
rz(3.1394203422020253) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.12662110611382) q[0];
rz(-2.7485087117386433) q[0];
ry(1.1993157978832798) q[1];
rz(3.139581223657284) q[1];
ry(-3.1371948307177435) q[2];
rz(0.9981012043496308) q[2];
ry(-0.008691527820850808) q[3];
rz(2.032427902551806) q[3];
ry(1.5710536859344082) q[4];
rz(3.141549545129477) q[4];
ry(-1.5718621730754598) q[5];
rz(-2.6183097181630215) q[5];
ry(1.5717736322999873) q[6];
rz(-3.1368825100060906) q[6];
ry(1.5681542249640463) q[7];
rz(2.6388562401881606) q[7];
ry(-1.570520356407217) q[8];
rz(-1.9473647521574566) q[8];
ry(1.5688637684335915) q[9];
rz(3.1414480782860945) q[9];
ry(1.5456566321586056) q[10];
rz(3.129750419502944) q[10];
ry(-1.6044112202746268) q[11];
rz(2.7865184447695825) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1207665077657887) q[0];
rz(-1.9137085191402843) q[0];
ry(1.6994138295503087) q[1];
rz(-0.7529387561115178) q[1];
ry(3.0361319229579897) q[2];
rz(-2.497283982052264) q[2];
ry(0.014809682949697396) q[3];
rz(1.7353598880211019) q[3];
ry(1.5710781139900307) q[4];
rz(-1.9093272373888803) q[4];
ry(3.1397642335379192) q[5];
rz(0.5233663673973759) q[5];
ry(-1.5699321962839026) q[6];
rz(-1.1688316916569679) q[6];
ry(1.5703294472153821) q[7];
rz(-1.859471726413801) q[7];
ry(-0.004292413127178347) q[8];
rz(0.4378344494667985) q[8];
ry(-1.5710256829568134) q[9];
rz(1.753983654249737) q[9];
ry(-0.022055836014423028) q[10];
rz(1.569495897245517) q[10];
ry(-2.9321376618462436) q[11];
rz(1.272934088363063) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1072144271401863) q[0];
rz(1.110584642734023) q[0];
ry(-0.47261822115757823) q[1];
rz(-2.463227694985234) q[1];
ry(-9.481391659501374e-05) q[2];
rz(2.6032637302316894) q[2];
ry(-1.5711537852566002) q[3];
rz(-2.3225607635150713) q[3];
ry(0.00012879945455540848) q[4];
rz(1.7649474297409165) q[4];
ry(-1.6541908581586335) q[5];
rz(-6.819946433722633e-05) q[5];
ry(0.00016540048074720204) q[6];
rz(1.1685004110031922) q[6];
ry(-3.1415666923975243) q[7];
rz(1.283306867045522) q[7];
ry(-0.006226319100353095) q[8];
rz(-2.429735138331465) q[8];
ry(-0.02508485626020202) q[9];
rz(2.9585642962158296) q[9];
ry(-1.5681313362044236) q[10];
rz(-1.5766628082586651) q[10];
ry(-1.572508741276262) q[11];
rz(-2.386173850380575) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.570274957665587) q[0];
rz(1.56451506573982) q[0];
ry(1.5710457022475068) q[1];
rz(1.6382750088719291) q[1];
ry(-0.0029468718458085514) q[2];
rz(-1.4626127462789829) q[2];
ry(3.1414758559730678) q[3];
rz(-1.4841322708098927) q[3];
ry(-3.1369207645995862) q[4];
rz(3.021634043268359) q[4];
ry(-1.5693224621451416) q[5];
rz(1.92910839191235) q[5];
ry(1.5704611836735938) q[6];
rz(0.00721863065170518) q[6];
ry(-1.5703627098687938) q[7];
rz(0.47361441692874195) q[7];
ry(3.1401534169345573) q[8];
rz(-0.792434706487042) q[8];
ry(-1.5674974759669444) q[9];
rz(0.0006920747911616483) q[9];
ry(0.19622092098950095) q[10];
rz(-0.007314458406324643) q[10];
ry(-0.0011855896646932318) q[11];
rz(-2.3333815224046592) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.10774904937253993) q[0];
rz(-0.8261607042696475) q[0];
ry(0.07186490829525255) q[1];
rz(2.2247567493446545) q[1];
ry(-1.5690506456572635) q[2];
rz(0.739144273975789) q[2];
ry(-0.006734253216596819) q[3];
rz(3.0234246358413177) q[3];
ry(-1.5657361443163402) q[4];
rz(0.7419275328430288) q[4];
ry(-0.00036027425998264235) q[5];
rz(0.35546226397060776) q[5];
ry(-1.562915286029888) q[6];
rz(0.7391433588479099) q[6];
ry(-0.0049491430814345705) q[7];
rz(-1.3260433038492208) q[7];
ry(-1.5810380806265538) q[8];
rz(-0.8101193154056805) q[8];
ry(-1.5782378738926188) q[9];
rz(-0.8252508647420759) q[9];
ry(-1.5767737002457265) q[10];
rz(-2.3726428130761126) q[10];
ry(1.5708622640106116) q[11];
rz(0.7080057039532788) q[11];