// Copyright 2021-2022, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Responses are stored on the
 * composed type: "ResponsePacket." See that file for details about what responses it stores. Output of this content
 * can be controlled by the responseCollector. Responses are defined as the following. . .
 *
 * - "Name" response: The name of the object that uses Voicing. Similar to the "Accessible Name" in web accessibility.
 * - "Object" response: The state information about the object that uses Voicing.
 * - "Context" response: The contextual changes that result from interaction with the Node that uses Voicing.
 * - "Hint" response: A supporting hint that guides the user toward a desired interaction with this Node.
 *
 * See ResponsePacket, as well as the property and setter documentation for each of these responses for more
 * information.
 *
 * Once this content is set, you can make a request to speak it using an UtteranceQueue with one of the provided
 * functions in this Trait. It is up to you to call one of these functions when you wish for speech to be made. The only
 * exception is on the 'focus' event. Every Node that composes Voicing will speak its responses by when it
 * receives focus.
 *
 * NOTE: At this time, you cannot use Voicing options and pass options through super(), instead you must call mutate
 * as a second statement. TODO: can we get rid of this stipulation? https://github.com/phetsims/scenery/issues/1340
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePacket, { ResponsePacketOptions } from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import { InteractiveHighlighting, Node, scenery, SceneryEvent, voicingUtteranceQueue } from '../../imports.js';
import optionize from '../../../../phet-core/js/optionize.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtteranceQueue',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener'
];

type VoicingOptions = {
  voicingNameResponse?: string | null,
  voicingObjectResponse?: string | null,
  voicingContextResponse?: string | null,
  voicingHintResponse?: string | null,
  voicingUtteranceQueue?: UtteranceQueue,
  voicingResponsePatternCollection?: ResponsePatternCollection,
  voicingIgnoreVoicingManagerProperties?: boolean,
  voicingFocusListener?: SceneryListenerFunction
}

type ResponseOptions = {
  utterance?: Utterance | null;
} & ResponsePacketOptions;

type SceneryListenerFunction = ( event: SceneryEvent ) => void;

type Constructor<T = {}> = new ( ...args: any[] ) => T;

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const Voicing = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {

  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose Voicing' );

  const InteractiveHighlightingClass = InteractiveHighlighting( Type, optionsArgPosition );

  // Unfortunately, nothing can be private or protected in this class, see https://github.com/phetsims/scenery/issues/1340#issuecomment-1020692592
  const VoicingClass = class extends InteractiveHighlightingClass {
    public voicingResponsePacket: ResponsePacket; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1340
    public _voicingUtteranceQueue: UtteranceQueue | null;
    public _voicingFocusListener: SceneryListenerFunction;
    public speakContentOnFocusListener: { focus: SceneryListenerFunction }; // TODO: use underscore so that there is a "private" convention. https://github.com/phetsims/scenery/issues/1340

    constructor( ...args: any[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as VoicingOptions;

      const voicingOptions = _.pick( providedOptions, VOICING_OPTION_KEYS );
      args[ optionsArgPosition ] = _.omit( providedOptions, VOICING_OPTION_KEYS );

      super( ...args );

      // @public {ResponsePacket} - ResponsePacket that holds all the supported responses to be Voiced
      this.voicingResponsePacket = new ResponsePacket();

      // @private {UtteranceQueue|null} - The utteranceQueue that responses for this Node will be spoken through.
      // By default (null), it will go through the singleton voicingUtteranceQueue, but you may need separate
      // UtteranceQueues for different areas of content in your application. For example, Voicing and
      // the default voicingUtteranceQueue may be disabled, but you could still want some speech to come through
      // while user is changing preferences or other settings.
      this._voicingUtteranceQueue = null;

      // @private {Function(event):} - called when this node is focused.
      this._voicingFocusListener = this.defaultFocusListener;

      // @private {Object} - Input listener that speaks content on focus. This is the only input listener added
      // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
      // response, and interaction hint.
      this.speakContentOnFocusListener = {
        focus: event => {
          this._voicingFocusListener( event );
        }
      };
      ( this as unknown as Node ).addInputListener( this.speakContentOnFocusListener );

      // @ts-ignore
      ( this as unknown as Node ).mutate( voicingOptions );
    }

    /**
     * Speak all responses assigned to this Node. Options allow you to override a responses for this particular
     * speech request. Each response is only spoken if the associated Property of responseCollector is true. If
     * all are Properties are false, nothing will be spoken.
     */
    voicingSpeakFullResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        nameResponse: this.voicingResponsePacket.nameResponse,
        objectResponse: this.voicingResponsePacket.objectResponse,
        contextResponse: this.voicingResponsePacket.contextResponse,
        hintResponse: this.voicingResponsePacket.hintResponse
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * Speak ONLY the provided responses that you pass in with options. This will NOT speak the name, object,
     * context, or hint responses assigned to this node by default. But it allows for clarity at usages so it is
     * clear that you are only requesting certain responses. If you want to speak all of the responses assigned
     * to this Node, use voicingSpeakFullResponse().
     *
     * Each response will only be spoken if the Properties of responseCollector are true. If all of those are false,
     * nothing will be spoken.
     */
    voicingSpeakResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        nameResponse: null,
        objectResponse: null,
        contextResponse: null,
        hintResponse: null
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the name response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.nameResponseEnabledProperty is false.
     */
    voicingSpeakNameResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        nameResponse: this.voicingResponsePacket.nameResponse
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
     */
    voicingSpeakObjectResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        objectResponse: this.voicingResponsePacket.objectResponse
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the context response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the name response may not be spoken if
     * responseCollector.contextResponseEnabledProperty is false.
     */
    voicingSpeakContextResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        contextResponse: this.voicingResponsePacket.contextResponse
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the hint response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the hint response may not be spoken if
     * responseCollector.hintResponseEnabledProperty is false.
     */
    voicingSpeakHintResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
        hintResponse: this.voicingResponsePacket.hintResponse
      }, providedOptions );

      this.collectAndSpeakResponse( options );
    }

    /**
     * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
     *
     * TODO: we want this to be @protected, https://github.com/phetsims/scenery/issues/1340
     * @public
     */
    collectAndSpeakResponse( providedOptions?: ResponseOptions ): void {
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {

        // {boolean} - whether or not this response should ignore the Properties of responseCollector
        ignoreProperties: this.voicingResponsePacket.ignoreProperties,

        // {Object} - collection of string patterns to use with responseCollector.collectResponses, see
        // ResponsePatternCollection for more information.
        responsePatternCollection: this.voicingResponsePacket.responsePatternCollection,

        // {Utterance|null} - The utterance to use if you want this response to be more controlled in the
        // UtteranceQueue.
        utterance: null
      }, providedOptions );

      // TODO: remove eslint-diable-line when AlertableDef is in a proper typescript file, https://github.com/phetsims/scenery/issues/1340
      let response: AlertableDef = responseCollector.collectResponses( options ); // eslint-disable-line no-undef

      if ( options.utterance ) {
        options.utterance.alert = response;
        response = options.utterance;
      }
      this.speakContent( response );
    }

    /**
     * Use the provided function to create content to speak in response to input. The content is then added to the
     * back of the voicing UtteranceQueue.
     *
     * TODO: we want this to be @protected, https://github.com/phetsims/scenery/issues/1340
     * @public
     *
     */
    // TODO: remove eslint-diable-line when AlertableDef is in a proper typescript file, https://github.com/phetsims/scenery/issues/1340
    speakContent( content: AlertableDef | null ): void { // eslint-disable-line no-undef

      // don't send to utteranceQueue if response is empty
      if ( content ) {
        const utteranceQueue = this.voicingUtteranceQueue || voicingUtteranceQueue;
        utteranceQueue.addToBack( content );
      }
    }

    /**
     * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
     * when the object receives input. When requesting speech, this will only be spoken if
     * responseCollector.nameResponsesEnabledProperty is set to true.
     */
    setVoicingNameResponse( response: string | null ): void {
      this.voicingResponsePacket.nameResponse = response;
    }

    set voicingNameResponse( response ) { this.setVoicingNameResponse( response ); }

    /**
     * Get the voicingNameResponse for this Node.
     */
    getVoicingNameResponse() {
      return this.voicingResponsePacket.nameResponse;
    }

    get voicingNameResponse() { return this.getVoicingNameResponse(); }

    /**
     * Set the object response for this Node. This is usually the state information associated with this Node, such
     * as its current input value. When requesting speech, this will only be heard when
     * responseCollector.objectResponsesEnabledProperty is set to true.
     */
    setVoicingObjectResponse( response: string | null ) {
      this.voicingResponsePacket.objectResponse = response;
    }

    set voicingObjectResponse( response ) { this.setVoicingObjectResponse( response ); }

    /**
     * Gets the object response for this Node.
     */
    getVoicingObjectResponse() {
      return this.voicingResponsePacket.objectResponse;
    }

    get voicingObjectResponse() { return this.getVoicingObjectResponse(); }

    /**
     * Set the context response for this Node. This is usually the content that describes what has happened in
     * the surrounding application in response to interaction with this Node. When requesting speech, this will
     * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
     */
    setVoicingContextResponse( response: string | null ) {
      this.voicingResponsePacket.contextResponse = response;
    }

    set voicingContextResponse( response ) { this.setVoicingContextResponse( response ); }

    /**
     * Gets the context response for this Node.
     */
    getVoicingContextResponse() {
      return this.voicingResponsePacket.contextResponse;
    }

    get voicingContextResponse() { return this.getVoicingContextResponse(); }

    /**
     * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
     * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
     * true.
     */
    setVoicingHintResponse( response: string | null ) {
      this.voicingResponsePacket.hintResponse = response;
    }

    set voicingHintResponse( response ) { this.setVoicingHintResponse( response ); }

    /**
     * Gets the hint response for this Node.
     */
    getVoicingHintResponse() {
      return this.voicingResponsePacket.hintResponse;
    }

    get voicingHintResponse() { return this.getVoicingHintResponse(); }

    /**
     * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
     * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
     * preferences.
     */
    setVoicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) {
      this.voicingResponsePacket.ignoreProperties = ignoreProperties;
    }

    set voicingIgnoreVoicingManagerProperties( ignoreProperties ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); }

    /**
     * Get whether or not responses are ignoring responseCollector Properties.
     */
    getVoicingIgnoreVoicingManagerProperties() {
      return this.voicingResponsePacket.ignoreProperties;
    }

    get voicingIgnoreVoicingManagerProperties() { return this.getVoicingIgnoreVoicingManagerProperties(); }

    /**
     * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
     * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
     * a collection of string patterns that are not the default.
     */
    setVoicingResponsePatternCollection( patterns: ResponsePatternCollection ) {
      assert && assert( patterns instanceof ResponsePatternCollection );
      this.voicingResponsePacket.responsePatternCollection = patterns;
    }

    set voicingResponsePatternCollection( patterns ) { this.setVoicingResponsePatternCollection( patterns ); }

    /**
     * Get the ResponsePatternCollection object that this Voicing Node is using to collect responses.
     */
    getVoicingResponsePatternCollection() {
      return this.voicingResponsePacket.responsePatternCollection;
    }

    get voicingResponsePatternCollection() { return this.getVoicingResponsePatternCollection(); }

    /**
     * Sets the utteranceQueue through which voicing associated with this Node will be spoken. By default,
     * the Display's voicingUtteranceQueue is used. But you can specify a different one if more complicated
     * management of voicing is necessary.
     */
    setVoicingUtteranceQueue( utteranceQueue: UtteranceQueue | null ) {
      this._voicingUtteranceQueue = utteranceQueue;
    }

    set voicingUtteranceQueue( utteranceQueue: UtteranceQueue | null ) { this.setVoicingUtteranceQueue( utteranceQueue ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingUtteranceQueue() {
      return this._voicingUtteranceQueue;
    }

    get voicingUtteranceQueue() { return this.getVoicingUtteranceQueue(); }

    /**
     * Called whenever this Node is focused.
     */
    setVoicingFocusListener( focusListener: SceneryListenerFunction ) {
      this._voicingFocusListener = focusListener;
    }

    set voicingFocusListener( focusListener: SceneryListenerFunction ) { this.setVoicingFocusListener( focusListener ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingFocusListener(): SceneryListenerFunction {
      return this._voicingFocusListener;
    }

    get voicingFocusListener() { return this.getVoicingFocusListener(); }


    /**
     * The default focus listener attached to this Node during initialization.
     */
    defaultFocusListener(): void {
      this.voicingSpeakFullResponse( {
        contextResponse: null
      } );
    }

    /**
     * Whether or not a Node composes Voicing.
     */
    get isVoicing() {
      return true;
    }

    /**
     * Detaches references that ensure this components of this Trait are eligible for garbage collection.
     * @public
     */
    dispose() {
      ( this as unknown as Node ).removeInputListener( this.speakContentOnFocusListener );

      super.dispose();
    }
  };

  /**
   * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
   * the order they will be evaluated.
   *
   * TODO: we want this to be @protected, https://github.com/phetsims/scenery/issues/1340
   * @public
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  VoicingClass.prototype._mutatorKeys = _.uniq( ( VoicingClass.prototype._mutatorKeys ? VoicingClass.prototype._mutatorKeys : [] ).concat( VOICING_OPTION_KEYS ) );
  return VoicingClass;
};

// @public
Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

scenery.register( 'Voicing', Voicing );
export default Voicing;
