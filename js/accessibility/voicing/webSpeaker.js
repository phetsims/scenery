// Copyright 2020-2021, University of Colorado Boulder

/**
 * Uses the Web Speech API to produce speech from the browser. This is a prototype, DO NOT USE IN PRODUCTION CODE.
 * There is no speech output until the webSpeaker has been initialized. Supported voices will depend on platform.
 * For each voice, you can customize the rate and pitch. Only one webSpeaker should be active at a time and so this
 * type is a singleton.
 *
 * @author Jesse Greenberg
 */

import BooleanProperty from '../../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import Emitter from '../../../../axon/js/Emitter.js';
import EnabledComponent from '../../../../axon/js/EnabledComponent.js';
import NumberProperty from '../../../../axon/js/NumberProperty.js';
import Property from '../../../../axon/js/Property.js';
import Range from '../../../../dot/js/Range.js';
import merge from '../../../../phet-core/js/merge.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import VoicingUtterance from '../../../../utterance-queue/js/VoicingUtterance.js';
import scenery from '../../scenery.js';
import globalKeyStateTracker from '../globalKeyStateTracker.js';
import KeyboardUtils from '../KeyboardUtils.js';

class WebSpeaker extends EnabledComponent {
  constructor() {

    super( {

      // initial value for the enabledProperty, false because speech should not happen until requested by user
      enabled: false,

      // phet-io
      phetioEnabledPropertyInstrumented: false
    } );

    // @public {null|SpeechSynthesisVoice}
    this.voiceProperty = new Property( null );

    // @public {NumberProperty} - controls the speaking rate of Web Speech
    this.voiceRateProperty = new NumberProperty( 1.0, { range: new Range( 0.75, 2 ) } );

    // {NumberProperty} - controls the pitch of the synth
    this.voicePitchProperty = new NumberProperty( 1.0, { range: new Range( 0.5, 2 ) } );

    // @public {Emitter} - emits events when the speaker starts/stops speaking, with the Utterance that is
    // either starting or stopping
    this.startSpeakingEmitter = new Emitter( { parameters: [ { valueType: Utterance } ] } );
    this.endSpeakingEmitter = new Emitter( { parameters: [ { valueType: Utterance } ] } );

    // @public {Emitter} - emits whenever the voices change for SpeechSynthesis
    this.voicesChangedEmitter = new Emitter();

    // @public - whether or not the synth is speaking - perhaps this should
    // replace the emitters above?
    this.speakingProperty = new BooleanProperty( false );

    // @private {SpeechSynthesis|null} - synth from Web Speech API that drives speech, defined on initialize
    this._synth = null;

    // @public {SpeechSynthesisVoice[]} - possible voices for Web Speech synthesis
    this.voices = [];

    // @public {boolean} - is the WebSpeaker initialized for use? This is prototypal so it isn't always initialized
    this.initialized = false;

    // @private {Property|DerivedProperty|null} - Controls whether or not speech is allowed with synthesis.
    // Null until initialized, and can be set by options to initialize().
    this._canSpeakProperty = null;

    // @private {function} - bound so we can link and unlink to this.canSpeakProperty when the webSpeaker becomes
    // initialized.
    this.boundHandleCanSpeakChange = this.handleCanSpeakChange.bind( this );

    // @private {Utterance} - A reference to the last utterance spoken, so we can determine
    // cancelling behavior when it is time to speak the next utterance. See VoicingUtterance options.
    this.previousUtterance = null;

    // @public {boolean} - a more interal way to disable speaking - the enabledProperty
    // can be set by the user and is publicly observable for other things - but if
    // you need to temporarily shut down speaking without changing that observable
    // you can set onHold to true to prevent all speaking. Useful in cases like
    // the ResetAllButton where you want to describe the reset without
    // any of the other changing Properties in that interaction
    this.onHold = false;

    // fixes a bug on Safari where the `start` and `end` Utterances don't fire! The
    // issue is (apparently) that Safari internally clears the reference to the
    // Utterance on speak which prevents it from firing these events at the right
    // time - fix borrowed from
    // https://stackoverflow.com/questions/23483990/speechsynthesis-api-onend-callback-not-working
    // Unfortunately, this also introduces a memory leak, we should be smarter about
    // clearing this, though it is a bit tricky since we don't have a way to know
    // when we are done with an utterance - see #215
    this.utterances = [];
  }

  /**
   * Indicate that the webSpeaker is ready for use, and attempt to populate voices (if they are ready yet). Adds
   * listeners that control speech.
   * @public
   */
  initialize( options ) {
    assert && assert( this.initialized === false, 'can only be initialized once' );
    assert && assert( this.isSpeechSynthesisSupported(), 'trying to initialize speech, but speech is not supported on this platform.' );

    options = merge( {

      // {BooleanProperty|DerivedProperty.<boolean>} - Controls whether speech is allowed with speech synthesis.
      // Combined into another DerivedProperty with this.enabledProperty so you don't have to use that as one
      // of the Properties that derive speechAllowedProperty, if you are passing in a DerivedProperty.
      speechAllowedProperty: new BooleanProperty( true )
    }, options );

    this._synth = window.speechSynthesis;

    this._canSpeakProperty = DerivedProperty.and( [ options.speechAllowedProperty, this.enabledProperty ] );
    this._canSpeakProperty.link( this.boundHandleCanSpeakChange );

    // browsers tend to generate the list of voices lazily, so the list of voices may be empty until speech is
    // first requested
    this.getSynth().onvoiceschanged = () => {
      this.populateVoices();
    };

    // try to populate voices immediately in case the browser populates them eagerly and we never get an
    // onvoiceschanged event
    this.populateVoices();

    // The control key will stop the synth from speaking if there is an active utterance. This key was decided because
    // most major screen readers will stop speech when this key is pressed
    globalKeyStateTracker.keyupEmitter.addListener( domEvent => {
      if ( KeyboardUtils.isControlKey( domEvent ) ) {
        this.cancel();
      }
    } );

    this.initialized = true;
  }

  /**
   * When we can no longer speak, cancel all speech to silence everything.
   * @private
   *
   * @param {boolean} canSpeak
   */
  handleCanSpeakChange( canSpeak ) {
    if ( !canSpeak ) { this.cancel(); }
  }

  /**
   * Update the list of voices available to the synth, and notify that the list has changed.
   * @private
   */
  populateVoices() {
    this.voices = this.getSynth().getVoices();
    this.voicesChangedEmitter.emit();
  }

  /**
   * Implements announce so the webSpeaker can be a source of output for utteranceQueue.
   * @public
   *
   * @param {Utterance} utterance
   */
  announce( utterance ) {
    let withCancel = true;
    if ( utterance instanceof VoicingUtterance ) {
      if ( this.previousUtterance && this.previousUtterance === utterance ) {
        withCancel = utterance.cancelSelf;
      }
      else {
        withCancel = utterance.cancelOther;
      }
    }

    this.speak( utterance, withCancel );
  }

  /**
   * Use speech synthesis to speak an utterance. No-op unless webSpeaker is initialized.
   * @public
   *
   * @param {Utterance} utterance
   * @param {boolean} withCancel - if true, any utterances remaining in the queue will be removed and this utterance
   *                               will take priority. Hopefully this works on all platforms, if it does not we
   *                               need to implement our own queing system.
   */
  speak( utterance, withCancel = true ) {
    if ( this.initialized && this._canSpeakProperty.value && !this.onHold ) {
      assert && assert( this.isSpeechSynthesisSupported(), 'trying to speak with speechSynthesis, but it is not supported on this platform' );

      // only cancel the previous alert if there is something new to speak
      if ( withCancel && utterance.alert ) {
        this.cancel();
      }

      // embeddding marks (for i18n) impact the output, strip before speaking
      const speechSynthUtterance = new SpeechSynthesisUtterance( stripEmbeddingMarks( utterance.getTextToAlert() ) );
      speechSynthUtterance.voice = this.voiceProperty.value;
      speechSynthUtterance.pitch = this.voicePitchProperty.value;
      speechSynthUtterance.rate = this.voiceRateProperty.value;

      // keep a reference to WebSpeechUtterances in Safari, so the browser doesn't dispose of it before firing, see #215
      this.utterances.push( speechSynthUtterance );

      // Keep this out of the start listener so that it can be synchrounous to the UtteranceQueue draining/announcing, see bug in https://github.com/phetsims/sun/issues/699#issuecomment-831529485
      this.previousUtterance = utterance;

      const startListener = () => {
        this.startSpeakingEmitter.emit( utterance );
        this.speakingProperty.set( true );
        speechSynthUtterance.removeEventListener( 'start', startListener );
      };

      const endListener = () => {

        this.endSpeakingEmitter.emit( utterance );
        this.speakingProperty.set( false );
        speechSynthUtterance.removeEventListener( 'end', endListener );

        // remove the reference to the SpeechSynthesisUtterance so we don't leak memory
        const indexOfUtterance = this.utterances.indexOf( speechSynthUtterance );
        if ( indexOfUtterance > -1 ) {
          this.utterances.splice( indexOfUtterance, 1 );
        }
      };

      speechSynthUtterance.addEventListener( 'start', startListener );
      speechSynthUtterance.addEventListener( 'end', endListener );

      // In Safari the `end` listener does not fire consistently, (especially after cancel)
      // but the error event does. In this case signify that speaking has ended.
      speechSynthUtterance.addEventListener( 'error', endListener );

      this.getSynth().speak( speechSynthUtterance );
    }
  }

  /**
   * Speak something initially and synchronously after some user interaction. This is helpful for a couple of cases:
   *   1) Browsers require that speech happen in response to some user interaction, with absolutely no delay.
   *   announce() is used with the utteranceQueue, which does not speak things instantly. Use this when speech is
   *   enabled, then use speak for all other usages.
   *   2) There are rare cases where we need to speak even when when canSpeakProperty is false (like speaking
   *   that voicing has been successfully turned off).
   * @public
   *
   * @param {string} utterThis
   */
  speakImmediately( utterThis ) {
    if ( this.initialized ) {
      assert && assert( this.isSpeechSynthesisSupported(), 'Trying to speak, but speech synthesis is not supported on this platform' );

      // embidding marks (for i18n) impact the output, strip before speaking
      const utterance = new SpeechSynthesisUtterance( stripEmbeddingMarks( utterThis ) );
      utterance.voice = this.voiceProperty.value;
      utterance.pitch = this.voicePitchProperty.value;
      utterance.rate = this.voiceRateProperty.value;

      // Keep this synchrounous as it pertains to the UtteranceQueue draining/announcing, see bug in https://github.com/phetsims/sun/issues/699#issuecomment-831529485
      this.previousUtterance = utterance;

      this.getSynth().speak( utterance );
    }
  }

  /**
   * Returns true if SpeechSynthesis is available on the window. This check is sufficient for all of
   * webSpeaker. On platforms where speechSynthesis is available, all features of it are available, with the
   * exception of the onvoiceschanged event in a couple of platforms. However, the listener can still be set
   * without issue on those platforms so we don't need to check for its existence. On those platforms, voices
   * are provided right on load.
   * @public
   *
   * @returns {boolean}
   */
  isSpeechSynthesisSupported() {
    return !!window.speechSynthesis && !!window.SpeechSynthesisUtterance;
  }

  /**
   * Returns a references to the SpeechSynthesis of the webSpeaker that is used to request speech with the Web
   * Speech API. Every references has a check to ensure that the synth is available.
   * @private
   *
   * @returns {null|SpeechSynthesis}
   */
  getSynth() {
    assert && assert( this.isSpeechSynthesisSupported(), 'Trying to use SpeechSynthesis, but it is not supported on this platform.' );
    return this._synth;
  }

  /**
   * Stops any current speech and removes all utterances in the queue internal to the SpeechSynthesis
   * (not the UtteranceQueue).
   * @public
   */
  cancel() {
    if ( this.initialized ) {
      this.getSynth().cancel();

      // cancel clears all utterances from the internal SpeechSynthsis queue so we should
      // clear all of our references as well
      this.utterances = [];
    }
  }
}

const webSpeaker = new WebSpeaker();

scenery.register( 'webSpeaker', webSpeaker );
export default webSpeaker;