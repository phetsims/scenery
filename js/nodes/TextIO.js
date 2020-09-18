// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for Text
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Denzell Barnett (PhET Interactive Simulations)
 */

import PropertyIO from '../../../axon/js/PropertyIO.js';
import merge from '../../../phet-core/js/merge.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import scenery from '../scenery.js';
import FontIO from '../util/FontIO.js';
import NodeProperty from '../util/NodeProperty.js';
import NodeIO from './NodeIO.js';

const TextIO = new IOType( 'TextIO', {
  valueType: scenery.Text,
  supertype: NodeIO,
  documentation: 'Text that is displayed in the simulation. TextIO has a nested PropertyIO.&lt;String&gt; for ' +
                 'the current string value.',
  createWrapper( text, phetioID ) {
    const superWrapper = this.supertype.createWrapper( text, phetioID );

    // this uses a sub Property adapter as described in https://github.com/phetsims/phet-io/issues/1326
    const textProperty = new NodeProperty( text, text.textProperty, 'text', merge( {

      // pick the following values from the parent Node
      phetioReadOnly: text.phetioReadOnly,
      phetioType: PropertyIO( StringIO ),

      tandem: text.tandem.createTandem( 'textProperty' ),
      phetioDocumentation: 'Property for the displayed text'
    }, text.phetioComponentOptions, text.phetioComponentOptions.textProperty ) );

    return {
      phetioObject: text,
      phetioID: phetioID,
      dispose: () => {
        superWrapper.dispose();
        textProperty.dispose();
      }
    };
  },
  methods: {
    setFontOptions: {
      returnType: VoidIO,
      parameterTypes: [ FontIO ],
      implementation: function( font ) {
        this.setFont( font );
      },
      documentation: 'Sets font options for this TextIO instance, e.g. {size: 16, weight: bold}. If increasing the font ' +
                     'size does not make the text size larger, you may need to increase the maxWidth of the TextIO also.',
      invocableForReadOnlyElements: false
    },

    getFontOptions: {
      returnType: FontIO,
      parameterTypes: [],
      implementation: function() {
        return this.getFont();
      },
      documentation: 'Gets font options for this TextIO instance as an object'
    },

    setMaxWidth: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      implementation: function( maxWidth ) {
        this.setMaxWidth( maxWidth );
      },
      documentation: 'Sets the maximum width of text box. ' +
                     'If the text width exceeds maxWidth, it is scaled down to fit.',
      invocableForReadOnlyElements: false
    },

    getMaxWidth: {
      returnType: NumberIO,
      parameterTypes: [],
      implementation: function() {
        return this.maxWidth;
      },
      documentation: 'Gets the maximum width of text box'
    }
  }
} );

scenery.register( 'TextIO', TextIO );
export default TextIO;