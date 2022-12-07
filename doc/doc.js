// Copyright 2022, University of Colorado Boulder

window.getCodeSnippet = func => {
  const pre = document.createElement( 'pre' );
  const code = document.createElement( 'code' );
  code.classList.add( 'language-typescript' );
  let js = func.toString().match( /\/\*START\*\/((.|\n)*)\/\*END\*\// )[ 1 ];
  let lines = js.split( '\n' );
  let minPadding = Number.POSITIVE_INFINITY;
  lines = lines.map( line => {
    const index = line.search( /[^ ]/ );
    if ( index === -1 ) {
      return '';
    }
    else {
      minPadding = Math.min( minPadding, index );
      return line;
    }
  } );
  while ( lines[ 0 ].length === 0 ) {
    lines.shift();
  }
  while ( lines[ lines.length - 1 ].length === 0 ) {
    lines.pop();
  }
  lines = lines.map( line => line.slice( minPadding ) );
  js = lines.join( '\n' );
  code.innerHTML = js;
  pre.appendChild( code );
  hljs.highlightElement( code ); // eslint-disable-line no-undef
  return pre;
};

window.generateAPIList = () => {
  const apiList = document.getElementById( 'apiList' );

  [ ...document.querySelectorAll( '.index' ) ].forEach( element => {
    const anchor = document.createElement( 'a' );
    anchor.classList.add( element.tagName === 'H2' ? 'navlink' : 'sublink' );
    anchor.href = `#${element.id}`;
    anchor.innerHTML = element.dataset.index ? element.dataset.index : element.innerHTML;
    apiList.appendChild( anchor );
    apiList.appendChild( document.createElement( 'br' ) );
  } );
};