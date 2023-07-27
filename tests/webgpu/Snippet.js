// Copyright 2023, University of Colorado Boulder


let globalSnippetIdCounter = 0;

// eslint-disable-next-line default-export-class-should-register-namespace
export default class Snippet {
  /**
   * Represents a piece of shader code with dependencies on other snippets. Supports serialization of only the
   * code that is needed.
   *
   * const a = new Snippet( 'A' )
   * const b = new Snippet( 'B', [a] )
   * const c = new Snippet( 'C', [a] )
   * const d = new Snippet( 'D', [b,c] )
   * d.toString() => "ABCD"
   * b.toString() => "AB"
   * c.toString() => "AC"
   *
   * @param {string} source
   * @param {Array.<Snippet>} dependencies
   */
  constructor( source, dependencies = [] ) {
    // @private {number}
    this.id = globalSnippetIdCounter++;

    // @private {string}
    this.source = source;

    // @private {Array.<Snippet>}
    this.dependencies = dependencies;
  }

  /**
   * Assuming no circular dependencies, this returns the entire required subprogram as a string.
   * usedSnippets is used for internal use, just call toString().
   * @public
   *
   * @param {Object} [usedSnippets] - Optional map from snippet ID => whether it was used.
   * @returns {boolean}
   */
  toString( usedSnippets = {} ) {
    let result = '';

    // if we have already been included, all of our dependencies have been included
    if ( usedSnippets[ this.id ] ) {
      return result;
    }

    if ( this.dependencies ) {
      for ( let i = 0; i < this.dependencies.length; i++ ) {
        result += this.dependencies[ i ].toString( usedSnippets );
      }
    }

    result += this.source;

    usedSnippets[ this.id ] = true;

    return result;
  }

  /**
   * Creates a snippet for a numeric constant from a given large-precision string.
   * @public
   *
   * @param {string} name
   * @param {string} value
   * @returns {Snippet}
   */
  static numericConstant( name, value ) {
    // Match WebGL handling
    return new Snippet( `#define ${name} ${value.substring( 0, 33 )}\n` );
  }

  /**
   * Turns a number into a GLSL-compatible float literal.
   * @public
   *
   * @param {number} n
   * @returns {string}
   */
  static toFloat( n ) {
    const s = n.toString();
    return ( s.indexOf( '.' ) < 0 && s.indexOf( 'e' ) < 0 && s.indexOf( 'E' ) < 0 ) ? ( s + '.0' ) : s;
  }
}
