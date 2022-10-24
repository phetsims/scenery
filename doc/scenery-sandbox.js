// Copyright 2022, University of Colorado Boulder

/**
 * Includes the code needed to create scenery-specific code sandboxes.
 *
 * Should include the following CSS/script references:
 *
 * <link rel="stylesheet" href="../../sherpa/lib/codemirror-5.52.2.min.css">
 * <link rel="stylesheet" href="../../sherpa/lib/codemirror-5.52.2.monokai.min.css">
 *
 * <script src="../../sherpa/lib/codemirror-5.52.2.min.js"></script>
 * <script src="../../sherpa/lib/codemirror-5.52.2.javascript.min.js"></script>
 *
 * AND the scenery-sandbox.css
 */

( () => {
  window.functionToJSSnippets = func => {
    const functionString = func.toString();
    let js = functionString.match( /\/\*START\*\/((.|\n)*)\/\*END\*\// )[ 1 ];
    let jsBefore = '';
    let jsAfter = '';
    if ( js.length !== functionString.length ) {
      jsBefore = functionString.slice( 0, functionString.indexOf( js ) );
      jsAfter = functionString.slice( functionString.indexOf( js ) + js.length );
    }
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
    return {
      js: js,
      jsBefore: jsBefore,
      jsAfter: jsAfter
    };
  };

  window.createSandbox = ( id, func, providedOptions ) => {

    const { js, jsBefore, jsAfter } = window.functionToJSSnippets( func );

    const options = phetCore.merge( { // eslint-disable-line no-undef
      jsBefore: jsBefore,
      jsAfter: jsAfter
    }, providedOptions );

    const parentElement = document.getElementById( id );

    const displayContainerElement = document.createElement( 'div' );
    parentElement.appendChild( displayContainerElement );

    const codeContainerElement = document.createElement( 'div' );
    parentElement.appendChild( codeContainerElement );

    const errorsContainerElement = document.createElement( 'div' );
    parentElement.appendChild( errorsContainerElement );
    errorsContainerElement.classList.add( 'errors' );

    const codeMirror = CodeMirror( codeContainerElement, { // eslint-disable-line no-undef
      lineNumbers: true,
      tabSize: 2,
      value: js,
      mode: 'javascript',
      theme: 'monokai',
      lineWrapping: true
    } );

    const container = new scenery.Node(); // eslint-disable-line no-undef
    const scene = new scenery.Node(); // eslint-disable-line no-undef

    container.addChild( scene );

    const display = new scenery.Display( container, { // eslint-disable-line no-undef
      width: 1,
      height: 1,
      accessibility: true,
      listenToOnlyElement: true,
      allowCSSHacks: false
    } );
    display.domElement.style.position = 'relative';

    const isDescendant = function( parent, child ) {
      let node = child;
      while ( node ) {
        if ( node === parent ) {
          return true;
        }

        // Traverse up to the parent
        node = node.parentNode;
      }

      // Go up until the root but couldn't find the `parent`
      return false;
    };

    window.addEventListener( 'keydown', event => {
      // if shift-enter is pressed
      if ( event.keyCode === 13 && event.shiftKey && isDescendant( document.getElementById( 'code' ), document.activeElement ) ) {
        run();

        event.preventDefault();
      }
    } );

    const stepEmitter = new axon.Emitter(); // eslint-disable-line no-undef

    display.updateOnRequestAnimationFrame( dt => {
      stepEmitter.emit( dt );

      const padding = 2;
      if ( scene.bounds.isValid() ) {
        const width = codeContainerElement.offsetWidth;
        scene.centerX = width / 2;
        scene.top = padding;
        display.width = width;
        display.height = Math.ceil( scene.bottom ) + padding;
      }
      displayContainerElement.style.height = `${Math.ceil( display.height )}px`;
    } );
    display.initializeEvents();

    displayContainerElement.appendChild( display._domElement );

    const run = async () => {
      const oldChildren = scene.children;
      scene.removeAllChildren();
      displayContainerElement.style.backgroundColor = 'transparent';
      scene.opacity = 1;
      errorsContainerElement.style.display = 'none';

      try {
        window[ `scene${id}` ] = scene;
        window[ `stepEmitter${id}` ] = stepEmitter;

        // Assumes it's in a function, differently from the sandbox
        const dataURI = `data:text/javascript;charset=utf-8,${Math.random()};(${options.jsBefore}${codeMirror.getValue()}${options.jsAfter})( window[ 'scene${id}' ], window[ 'stepEmitter${id}' ] )`;  // eslint-disable-line bad-sim-text

        await import( dataURI );
      }
      catch( e ) {
        console.error( e );
        scene.children = oldChildren;
        displayContainerElement.style.backgroundColor = 'rgba(255,0,0,0.2)';
        scene.opacity = 0.5;

        errorsContainerElement.style.display = 'block';
        errorsContainerElement.innerHTML = `<pre>${e}</pre>`;
      }
    };

    codeMirror.on( 'change', editor => run && run() );

    run();
  };
} )();
