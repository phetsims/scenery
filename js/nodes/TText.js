// Copyright 2016-2017, University of Colorado Boulder

/**
 * Wrapper type for scenery's Text node.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  var TFont = require( 'SCENERY/util/TFont' );
  var TNode = require( 'SCENERY/nodes/TNode' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var phetioEvents = require( 'ifphetio!PHET_IO/phetioEvents' );
  var TNumber = require( 'ifphetio!PHET_IO/types/TNumber' );
  var TString = require( 'ifphetio!PHET_IO/types/TString' );
  var TVoid = require( 'ifphetio!PHET_IO/types/TVoid' );
  var TFunctionWrapper = require( 'ifphetio!PHET_IO/types/TFunctionWrapper' );

  /**
   * Wrapper type for scenery's Text node.
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TText( text, phetioID ) {
    TNode.call( this, text, phetioID );
    assertInstanceOf( text, phet.scenery.Text );
    text.on( 'text', function( oldText, newText ) {
      phetioEvents.trigger( 'model', phetioID, TText, 'textChanged', {
        oldText: oldText,
        newText: newText
      } );
    } );
  }

  phetioInherit( TNode, 'TText', TText, {

    addTextChangedListener: {
      returnType: TVoid,
      parameterTypes: [ TFunctionWrapper( TVoid, [ TString ] ) ],
      implementation: function( listener ) {
        this.instance.on( 'text', function( oldText, newText ) {
          listener( newText );
        } );
      },
      documentation: 'Add a listener for when the text has changed.'
    },

    setText: {
      returnType: TVoid,
      parameterTypes: [ TString ],
      implementation: function( text ) {
        this.instance.text = text;
      },
      documentation: 'Set the text content'
    },

    getText: {
      returnType: TString,
      parameterTypes: [],
      implementation: function() {
        return this.instance.text;
      },
      documentation: 'Get the text content'
    },

    setFontOptions: {
      returnType: TVoid,
      parameterTypes: [ TFont ],
      implementation: function( font ) {
        this.instance.setFont( font );
      },
      documentation: 'Set font options for this TText instance, e.g. {size: 16, weight: bold}'
    },

    getFontOptions: {
      returnType: TFont,
      parameterTypes: [],
      implementation: function() {
        return this.instance.getFont();
      },
      documentation: 'Get font options for this TText instance as an object'
    },

    setMaxWidth: {
      returnType: TVoid,
      parameterTypes: [ TNumber() ],
      implementation: function( maxWidth ) {
        this.instance.setMaxWidth( maxWidth );
      },
      documentation: 'Set maximum width of text box in px. ' +
                     'If text is wider than maxWidth at its default font size, it is scaled down to fit.'
    },

    getMaxWidth: {
      returnType: TNumber(),
      parameterTypes: [],
      implementation: function() {
        return this.instance.maxWidth;
      },
      documentation: 'Get maximum width of text box in px'
    }
  }, {
    documentation: 'The tandem wrapper type for the scenery Text node',
    events: [ 'textChanged' ]
  } );

  scenery.register( 'TText', TText );

  return TText;
} );