// Copyright 2016-2017, University of Colorado Boulder

/**
 * IO type for scenery's Text node.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );
  var FontIO = require( 'SCENERY/util/FontIO' );
  var NodeIO = require( 'SCENERY/nodes/NodeIO' );

  // phet-io modules
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  var phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );
  var FunctionIO = require( 'ifphetio!PHET_IO/types/FunctionIO' );
  var NumberIO = require( 'ifphetio!PHET_IO/types/NumberIO' );
  var StringIO = require( 'ifphetio!PHET_IO/types/StringIO' );
  var VoidIO = require( 'ifphetio!PHET_IO/types/VoidIO' );

  /**
   * IO type for scenery's Text node.
   * @param {Text} text
   * @param {string} phetioID
   * @constructor
   */
  function TextIO( text, phetioID ) {
    assert && assertInstanceOf( text, phet.scenery.Text );
    NodeIO.call( this, text, phetioID );
  }

  phetioInherit( NodeIO, 'TextIO', TextIO, {

    addTextChangedListener: {
      returnType: VoidIO,
      parameterTypes: [ FunctionIO( VoidIO, [ StringIO ] ) ],
      implementation: function( listener ) {
        this.instance.on( 'text', function( oldText, newText ) {
          listener( newText );
        } );
      },
      documentation: 'Add a listener for when the text has changed.'
    },

    setText: {
      returnType: VoidIO,
      parameterTypes: [ StringIO ],
      implementation: function( text ) {
        this.instance.text = text;
      },
      documentation: 'Set the text content'
    },

    getText: {
      returnType: StringIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.text;
      },
      documentation: 'Get the text content'
    },

    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ FontIO ],
      implementation: function( font ) {
        this.instance.setFont( font );
      },
      documentation: 'Set font options for this TextIO instance, e.g. {size: 16, weight: bold}'
    },

    getFontOptions: {
      returnType: FontIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.getFont();
      },
      documentation: 'Get font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( maxWidth ) {
        this.instance.setMaxWidth( maxWidth );
      },
      documentation: 'Set maximum width of text box in px. ' +
                     'If text is wider than maxWidth at its default font size, it is scaled down to fit.'
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function() {
        return this.instance.maxWidth;
      },
      documentation: 'Get maximum width of text box in px'
    }
  }, {
    documentation: 'The tandem IO type for the scenery Text node',
    events: [ 'changed' ]
  } );

  scenery.register( 'TextIO', TextIO );

  return TextIO;
} );
